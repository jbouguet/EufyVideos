#!/usr/bin/env python3
"""
Cluster Labeling and Mega-Cluster Grouping Tool

This tool provides functionality for manually labeling person clusters and 
grouping them into mega-clusters representing the same individual. It serves 
as the bridge between conservative clustering and the final refined model.

Key Features:
- Load and display cluster results with thumbnails
- Assign human-readable labels to clusters
- Group clusters into mega-clusters (same person)
- Interactive management of cluster relationships
- Export labeled data for training refined models
- Support for manual corrections and adjustments

Workflow:
1. Load conservative clustering results
2. Review cluster grids (from visual inspection)
3. Assign labels to individual clusters
4. Group related clusters into mega-clusters
5. Export training data for refined model

Usage:
    python cluster_labeling_tool.py

The tool creates a comprehensive labeling system that enables:
- Manual cluster verification and labeling
- Mega-cluster creation for same-person grouping
- Training data generation for improved auto-labeling
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from enhanced_person_clustering import EnhancedPersonCluster, EnhancedPersonClusterer
from person_embedding import PersonEmbedding
from logging_config import create_logger

logger = create_logger(__name__)


@dataclass
class ClusterLabel:
    """Label information for a cluster."""
    
    cluster_id: int
    label: str  # Human-readable label like "Person_A", "Unknown_01", etc.
    confidence: float = 1.0  # Manual labels have high confidence
    notes: str = ""
    verified: bool = False  # Has this cluster been manually verified?
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MegaCluster:
    """A mega-cluster representing the same person across multiple clusters."""
    
    mega_id: str
    person_name: str  # e.g., "Person_A", "John_Doe", "Unknown_01"
    cluster_ids: List[int] = field(default_factory=list)
    confidence: float = 1.0  # Manual grouping has high confidence
    notes: str = ""
    created_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_cluster(self, cluster_id: int):
        """Add a cluster to this mega-cluster."""
        if cluster_id not in self.cluster_ids:
            self.cluster_ids.append(cluster_id)
            self.modified_timestamp = datetime.now().isoformat()
    
    def remove_cluster(self, cluster_id: int):
        """Remove a cluster from this mega-cluster."""
        if cluster_id in self.cluster_ids:
            self.cluster_ids.remove(cluster_id)
            self.modified_timestamp = datetime.now().isoformat()
    
    @property
    def total_embeddings(self) -> int:
        """Total embeddings across all clusters in this mega-cluster."""
        # This would need cluster data to calculate
        return len(self.cluster_ids) * 2  # Placeholder
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'mega_id': self.mega_id,
            'person_name': self.person_name,
            'cluster_ids': self.cluster_ids,
            'confidence': self.confidence,
            'notes': self.notes,
            'created_timestamp': self.created_timestamp,
            'modified_timestamp': self.modified_timestamp,
            'total_clusters': len(self.cluster_ids)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MegaCluster':
        """Create from dictionary (JSON deserialization)."""
        return cls(
            mega_id=data['mega_id'],
            person_name=data['person_name'],
            cluster_ids=data.get('cluster_ids', []),
            confidence=data.get('confidence', 1.0),
            notes=data.get('notes', ''),
            created_timestamp=data.get('created_timestamp', datetime.now().isoformat()),
            modified_timestamp=data.get('modified_timestamp', datetime.now().isoformat())
        )


class ClusterLabelingManager:
    """
    Manager for cluster labeling and mega-cluster grouping operations.
    
    Handles the complete workflow from conservative clusters to labeled
    mega-clusters ready for training refined models.
    """
    
    def __init__(self, 
                 clusters_file: str,
                 output_dir: str,
                 visual_inspection_dir: Optional[str] = None):
        """
        Initialize the cluster labeling manager.
        
        Args:
            clusters_file: Path to enhanced_clusters.json
            output_dir: Directory to save labeling results
            visual_inspection_dir: Directory with visual inspection grids
        """
        self.clusters_file = clusters_file
        self.output_dir = output_dir
        self.visual_inspection_dir = visual_inspection_dir
        
        # Data structures
        self.clusters: List[EnhancedPersonCluster] = []
        self.cluster_labels: Dict[int, ClusterLabel] = {}
        self.mega_clusters: Dict[str, MegaCluster] = {}
        self.cluster_to_mega: Dict[int, str] = {}  # cluster_id -> mega_id
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # File paths
        self.labels_file = os.path.join(output_dir, "cluster_labels.json")
        self.mega_clusters_file = os.path.join(output_dir, "mega_clusters.json")
        self.training_data_file = os.path.join(output_dir, "training_data.json")
        
        logger.info(f"ClusterLabelingManager initialized:")
        logger.info(f"  Clusters file: {clusters_file}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Visual inspection: {visual_inspection_dir}")
    
    def load_clusters(self):
        """Load enhanced clusters from JSON file."""
        logger.info("Loading enhanced clusters...")
        
        with open(self.clusters_file, 'r') as f:
            data = json.load(f)
        
        self.clusters = []
        for cluster_data in data['clusters']:
            cluster = EnhancedPersonCluster(
                cluster_id=cluster_data['cluster_id'],
                cluster_size=cluster_data['cluster_size'],
                avg_similarity=cluster_data['avg_similarity'],
                min_similarity=cluster_data['min_similarity'],
                quality_score=cluster_data['quality_score']
            )
            
            # Load embeddings
            for emb_data in cluster_data['embeddings']:
                embedding = PersonEmbedding.from_dict(emb_data)
                cluster.embeddings.append(embedding)
            
            self.clusters.append(cluster)
        
        logger.info(f"Loaded {len(self.clusters)} clusters for labeling")
    
    def load_existing_labels(self):
        """Load existing cluster labels if they exist."""
        if os.path.exists(self.labels_file):
            logger.info("Loading existing cluster labels...")
            with open(self.labels_file, 'r') as f:
                labels_data = json.load(f)
            
            self.cluster_labels = {}
            for label_data in labels_data['labels']:
                label = ClusterLabel(
                    cluster_id=label_data['cluster_id'],
                    label=label_data['label'],
                    confidence=label_data.get('confidence', 1.0),
                    notes=label_data.get('notes', ''),
                    verified=label_data.get('verified', False),
                    timestamp=label_data.get('timestamp', datetime.now().isoformat())
                )
                self.cluster_labels[label.cluster_id] = label
            
            logger.info(f"Loaded {len(self.cluster_labels)} existing labels")
    
    def load_existing_mega_clusters(self):
        """Load existing mega-clusters if they exist."""
        if os.path.exists(self.mega_clusters_file):
            logger.info("Loading existing mega-clusters...")
            with open(self.mega_clusters_file, 'r') as f:
                mega_data = json.load(f)
            
            self.mega_clusters = {}
            self.cluster_to_mega = {}
            
            for mega_data_item in mega_data['mega_clusters']:
                mega = MegaCluster.from_dict(mega_data_item)
                self.mega_clusters[mega.mega_id] = mega
                
                # Update cluster-to-mega mapping
                for cluster_id in mega.cluster_ids:
                    self.cluster_to_mega[cluster_id] = mega.mega_id
            
            logger.info(f"Loaded {len(self.mega_clusters)} existing mega-clusters")
    
    def assign_cluster_label(self, cluster_id: int, label: str, notes: str = "", verified: bool = True):
        """
        Assign a label to a cluster.
        
        Args:
            cluster_id: ID of the cluster to label
            label: Human-readable label
            notes: Optional notes about the cluster
            verified: Whether this cluster has been manually verified
        """
        cluster_label = ClusterLabel(
            cluster_id=cluster_id,
            label=label,
            notes=notes,
            verified=verified,
            timestamp=datetime.now().isoformat()
        )
        
        self.cluster_labels[cluster_id] = cluster_label
        logger.info(f"Assigned label '{label}' to cluster {cluster_id}")
    
    def create_mega_cluster(self, person_name: str, cluster_ids: List[int], notes: str = "") -> str:
        """
        Create a new mega-cluster.
        
        Args:
            person_name: Name for this person (e.g., "Person_A", "John_Doe")
            cluster_ids: List of cluster IDs to include
            notes: Optional notes about this person
            
        Returns:
            The mega-cluster ID
        """
        mega_id = str(uuid.uuid4())
        
        # Remove clusters from existing mega-clusters
        for cluster_id in cluster_ids:
            if cluster_id in self.cluster_to_mega:
                old_mega_id = self.cluster_to_mega[cluster_id]
                self.mega_clusters[old_mega_id].remove_cluster(cluster_id)
                if not self.mega_clusters[old_mega_id].cluster_ids:
                    # Remove empty mega-cluster
                    del self.mega_clusters[old_mega_id]
        
        # Create new mega-cluster
        mega = MegaCluster(
            mega_id=mega_id,
            person_name=person_name,
            cluster_ids=cluster_ids.copy(),
            notes=notes
        )
        
        self.mega_clusters[mega_id] = mega
        
        # Update cluster-to-mega mapping
        for cluster_id in cluster_ids:
            self.cluster_to_mega[cluster_id] = mega_id
        
        logger.info(f"Created mega-cluster '{person_name}' with {len(cluster_ids)} clusters")
        return mega_id
    
    def add_cluster_to_mega(self, cluster_id: int, mega_id: str):
        """Add a cluster to an existing mega-cluster."""
        if mega_id not in self.mega_clusters:
            logger.error(f"Mega-cluster {mega_id} not found")
            return
        
        # Remove from old mega-cluster if exists
        if cluster_id in self.cluster_to_mega:
            old_mega_id = self.cluster_to_mega[cluster_id]
            self.mega_clusters[old_mega_id].remove_cluster(cluster_id)
        
        # Add to new mega-cluster
        self.mega_clusters[mega_id].add_cluster(cluster_id)
        self.cluster_to_mega[cluster_id] = mega_id
        
        logger.info(f"Added cluster {cluster_id} to mega-cluster {mega_id}")
    
    def remove_cluster_from_mega(self, cluster_id: int):
        """Remove a cluster from its mega-cluster."""
        if cluster_id in self.cluster_to_mega:
            mega_id = self.cluster_to_mega[cluster_id]
            self.mega_clusters[mega_id].remove_cluster(cluster_id)
            del self.cluster_to_mega[cluster_id]
            
            # Remove empty mega-cluster
            if not self.mega_clusters[mega_id].cluster_ids:
                del self.mega_clusters[mega_id]
            
            logger.info(f"Removed cluster {cluster_id} from mega-cluster {mega_id}")
    
    def save_labels(self):
        """Save cluster labels to file."""
        labels_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_labels': len(self.cluster_labels),
                'total_clusters': len(self.clusters)
            },
            'labels': [
                {
                    'cluster_id': label.cluster_id,
                    'label': label.label,
                    'confidence': label.confidence,
                    'notes': label.notes,
                    'verified': label.verified,
                    'timestamp': label.timestamp
                }
                for label in self.cluster_labels.values()
            ]
        }
        
        with open(self.labels_file, 'w') as f:
            json.dump(labels_data, f, indent=2)
        
        logger.info(f"Saved {len(self.cluster_labels)} labels to {self.labels_file}")
    
    def save_mega_clusters(self):
        """Save mega-clusters to file."""
        mega_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_mega_clusters': len(self.mega_clusters),
                'total_clustered': len(self.cluster_to_mega),
                'total_unclustered': len(self.clusters) - len(self.cluster_to_mega)
            },
            'mega_clusters': [mega.to_dict() for mega in self.mega_clusters.values()]
        }
        
        with open(self.mega_clusters_file, 'w') as f:
            json.dump(mega_data, f, indent=2)
        
        logger.info(f"Saved {len(self.mega_clusters)} mega-clusters to {self.mega_clusters_file}")
    
    def generate_training_data(self) -> Dict[str, Any]:
        """
        Generate training data for refined model.
        
        Returns:
            Dictionary containing training data with positive/negative pairs
        """
        logger.info("Generating training data for refined model...")
        
        # Collect all embeddings by mega-cluster
        mega_embeddings = {}
        for mega in self.mega_clusters.values():
            embeddings = []
            for cluster_id in mega.cluster_ids:
                cluster = next((c for c in self.clusters if c.cluster_id == cluster_id), None)
                if cluster:
                    embeddings.extend(cluster.embeddings)
            mega_embeddings[mega.mega_id] = {
                'person_name': mega.person_name,
                'embeddings': embeddings,
                'total_embeddings': len(embeddings)
            }
        
        # Generate positive pairs (same person)
        positive_pairs = []
        for mega_id, data in mega_embeddings.items():
            embeddings = data['embeddings']
            person_name = data['person_name']
            
            # Create pairs within the same mega-cluster
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    positive_pairs.append({
                        'embedding1_id': f"{embeddings[i].video_filename}_{embeddings[i].frame_number}_{embeddings[i].track_id}",
                        'embedding2_id': f"{embeddings[j].video_filename}_{embeddings[j].frame_number}_{embeddings[j].track_id}",
                        'embedding1': embeddings[i].to_dict(),
                        'embedding2': embeddings[j].to_dict(),
                        'label': 1,  # Same person
                        'person_name': person_name,
                        'mega_cluster_id': mega_id
                    })
        
        # Generate negative pairs (different people)
        negative_pairs = []
        mega_list = list(mega_embeddings.items())
        
        for i in range(len(mega_list)):
            for j in range(i + 1, len(mega_list)):
                mega_id1, data1 = mega_list[i]
                mega_id2, data2 = mega_list[j]
                
                # Sample embeddings from each mega-cluster
                emb1_list = data1['embeddings'][:5]  # Limit for efficiency
                emb2_list = data2['embeddings'][:5]
                
                for emb1 in emb1_list:
                    for emb2 in emb2_list:
                        negative_pairs.append({
                            'embedding1_id': f"{emb1.video_filename}_{emb1.frame_number}_{emb1.track_id}",
                            'embedding2_id': f"{emb2.video_filename}_{emb2.frame_number}_{emb2.track_id}",
                            'embedding1': emb1.to_dict(),
                            'embedding2': emb2.to_dict(),
                            'label': 0,  # Different people
                            'person1_name': data1['person_name'],
                            'person2_name': data2['person_name'],
                            'mega_cluster1_id': mega_id1,
                            'mega_cluster2_id': mega_id2
                        })
        
        # Limit negative pairs to balance dataset
        max_negative = min(len(positive_pairs) * 2, len(negative_pairs))
        negative_pairs = negative_pairs[:max_negative]
        
        training_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_mega_clusters': len(self.mega_clusters),
                'positive_pairs': len(positive_pairs),
                'negative_pairs': len(negative_pairs),
                'total_pairs': len(positive_pairs) + len(negative_pairs)
            },
            'mega_clusters_summary': {
                mega_id: {
                    'person_name': data['person_name'],
                    'total_embeddings': data['total_embeddings']
                }
                for mega_id, data in mega_embeddings.items()
            },
            'positive_pairs': positive_pairs,
            'negative_pairs': negative_pairs
        }
        
        # Save training data
        with open(self.training_data_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Generated training data:")
        logger.info(f"  Positive pairs: {len(positive_pairs)}")
        logger.info(f"  Negative pairs: {len(negative_pairs)}")
        logger.info(f"  Saved to: {self.training_data_file}")
        
        return training_data
    
    def create_labeling_report(self) -> str:
        """Create a comprehensive labeling report."""
        report_path = os.path.join(self.output_dir, "labeling_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("CLUSTER LABELING AND MEGA-CLUSTERING REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total clusters: {len(self.clusters)}\n")
            f.write(f"Labeled clusters: {len(self.cluster_labels)}\n")
            f.write(f"Mega-clusters: {len(self.mega_clusters)}\n")
            f.write(f"Clustered: {len(self.cluster_to_mega)}\n")
            f.write(f"Unclustered: {len(self.clusters) - len(self.cluster_to_mega)}\n\n")
            
            # Mega-cluster summary
            f.write("MEGA-CLUSTERS SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            for mega in sorted(self.mega_clusters.values(), key=lambda x: len(x.cluster_ids), reverse=True):
                total_embeddings = sum(
                    c.cluster_size for c in self.clusters 
                    if c.cluster_id in mega.cluster_ids
                )
                
                f.write(f"\n{mega.person_name} (ID: {mega.mega_id[:8]}...):\n")
                f.write(f"  Clusters: {len(mega.cluster_ids)}\n")
                f.write(f"  Total embeddings: {total_embeddings}\n")
                f.write(f"  Cluster IDs: {sorted(mega.cluster_ids)}\n")
                if mega.notes:
                    f.write(f"  Notes: {mega.notes}\n")
            
            # Unclustered clusters
            unclustered = [c.cluster_id for c in self.clusters if c.cluster_id not in self.cluster_to_mega]
            if unclustered:
                f.write(f"\nUNCLUSTERED CLUSTERS: {sorted(unclustered)}\n")
            
            # Labels summary
            f.write(f"\nLABELS SUMMARY\n")
            f.write("-" * 20 + "\n")
            
            verified_count = sum(1 for label in self.cluster_labels.values() if label.verified)
            f.write(f"Verified labels: {verified_count}/{len(self.cluster_labels)}\n")
            
            for label in sorted(self.cluster_labels.values(), key=lambda x: x.cluster_id):
                f.write(f"Cluster {label.cluster_id}: {label.label}")
                if label.notes:
                    f.write(f" ({label.notes})")
                if not label.verified:
                    f.write(" [UNVERIFIED]")
                f.write("\n")
        
        logger.info(f"Labeling report saved: {report_path}")
        return report_path
    
    def interactive_labeling_session(self):
        """Run an interactive labeling session (command-line interface)."""
        logger.info("=" * 60)
        logger.info("INTERACTIVE CLUSTER LABELING SESSION")
        logger.info("=" * 60)
        
        print("\nAvailable commands:")
        print("  label <cluster_id> <label> [notes] - Label a cluster")
        print("  create <person_name> <cluster_ids> - Create mega-cluster")
        print("  add <cluster_id> <mega_id> - Add cluster to mega-cluster")
        print("  remove <cluster_id> - Remove cluster from mega-cluster")
        print("  show clusters - Show all clusters")
        print("  show megas - Show all mega-clusters")
        print("  show <cluster_id> - Show cluster details")
        print("  save - Save current progress")
        print("  quit - Save and exit")
        
        while True:
            try:
                command = input("\n> ").strip().split()
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    self.save_labels()
                    self.save_mega_clusters()
                    print("Progress saved. Goodbye!")
                    break
                
                elif cmd == 'save':
                    self.save_labels()
                    self.save_mega_clusters()
                    print("Progress saved.")
                
                elif cmd == 'label' and len(command) >= 3:
                    cluster_id = int(command[1])
                    label = command[2]
                    notes = ' '.join(command[3:]) if len(command) > 3 else ""
                    self.assign_cluster_label(cluster_id, label, notes)
                    print(f"Labeled cluster {cluster_id} as '{label}'")
                
                elif cmd == 'create' and len(command) >= 3:
                    person_name = command[1]
                    cluster_ids = [int(x) for x in command[2].split(',')]
                    notes = ' '.join(command[3:]) if len(command) > 3 else ""
                    mega_id = self.create_mega_cluster(person_name, cluster_ids, notes)
                    print(f"Created mega-cluster '{person_name}' with ID {mega_id[:8]}...")
                
                elif cmd == 'show':
                    if len(command) == 2:
                        if command[1] == 'clusters':
                            print(f"\nClusters ({len(self.clusters)}):")
                            for cluster in sorted(self.clusters, key=lambda x: x.cluster_id):
                                label = self.cluster_labels.get(cluster.cluster_id)
                                label_text = f" [{label.label}]" if label else " [unlabeled]"
                                mega_text = ""
                                if cluster.cluster_id in self.cluster_to_mega:
                                    mega_id = self.cluster_to_mega[cluster.cluster_id]
                                    mega = self.mega_clusters[mega_id]
                                    mega_text = f" -> {mega.person_name}"
                                print(f"  {cluster.cluster_id}: size={cluster.cluster_size}, "
                                     f"quality={cluster.quality_score:.3f}{label_text}{mega_text}")
                        
                        elif command[1] == 'megas':
                            print(f"\nMega-clusters ({len(self.mega_clusters)}):")
                            for mega in sorted(self.mega_clusters.values(), key=lambda x: x.person_name):
                                print(f"  {mega.person_name} ({mega.mega_id[:8]}...): "
                                     f"{len(mega.cluster_ids)} clusters {mega.cluster_ids}")
                        
                        else:
                            cluster_id = int(command[1])
                            cluster = next((c for c in self.clusters if c.cluster_id == cluster_id), None)
                            if cluster:
                                print(f"\nCluster {cluster_id}:")
                                print(f"  Size: {cluster.cluster_size}")
                                print(f"  Quality: {cluster.quality_score:.3f}")
                                print(f"  Avg similarity: {cluster.avg_similarity:.3f}")
                                
                                label = self.cluster_labels.get(cluster_id)
                                if label:
                                    print(f"  Label: {label.label}")
                                    if label.notes:
                                        print(f"  Notes: {label.notes}")
                                
                                if cluster_id in self.cluster_to_mega:
                                    mega_id = self.cluster_to_mega[cluster_id]
                                    mega = self.mega_clusters[mega_id]
                                    print(f"  Mega-cluster: {mega.person_name}")
                            else:
                                print(f"Cluster {cluster_id} not found")
                
                else:
                    print("Invalid command. Type 'quit' to exit.")
            
            except (ValueError, IndexError) as e:
                print(f"Error: {e}")
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit properly.")


def main():
    """Run cluster labeling tool."""
    
    # Configuration
    clusters_file = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/enhanced_clustering/enhanced_clusters.json"
    output_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/cluster_labeling"
    visual_inspection_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/visual_inspection"
    
    try:
        # Create labeling manager
        manager = ClusterLabelingManager(clusters_file, output_dir, visual_inspection_dir)
        
        # Load data
        manager.load_clusters()
        manager.load_existing_labels()
        manager.load_existing_mega_clusters()
        
        # Create initial report
        manager.create_labeling_report()
        
        logger.info("=" * 60)
        logger.info("CLUSTER LABELING TOOL READY")
        logger.info("=" * 60)
        logger.info(f"Loaded {len(manager.clusters)} clusters")
        logger.info(f"Existing labels: {len(manager.cluster_labels)}")
        logger.info(f"Existing mega-clusters: {len(manager.mega_clusters)}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Review cluster grids in visual inspection directory")
        logger.info("2. Use interactive session to label and group clusters")
        logger.info("3. Generate training data for refined model")
        
        # Ask user what to do
        print("\nOptions:")
        print("1. Interactive labeling session")
        print("2. Generate training data (if labeling is complete)")
        print("3. Create report and exit")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == '1':
            manager.interactive_labeling_session()
        elif choice == '2':
            manager.generate_training_data()
            manager.create_labeling_report()
        elif choice == '3':
            manager.create_labeling_report()
        else:
            print("Invalid choice. Creating report and exiting.")
            manager.create_labeling_report()
        
        return 0
        
    except Exception as e:
        logger.error(f"Cluster labeling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())