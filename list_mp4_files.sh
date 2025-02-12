#!/bin/bash

# Define the root directory
root_directory="/Users/jbouguet/Documents/EufySecurityVideos/record/extra"

# Flag for executing file renaming
execute_renaming=false

# Define the path to the devices.csv file
devices_csv_path="$(dirname "$0")/devices.csv"

# Load devices from the devices.csv file
# Requirements for devices.csv:
# 1. File format: CSV with two columns - "Serial" and "Device"
# 2. File location: Same directory as this config.py file
# 3. Order: The order of devices in the CSV file is preserved and used for plotting
# 4. The header row Serial,Device is expected in the CSV file
load_devices_from_csv() {
	local csv_file="$1"
	local line_number=0
	local devices=()
	local seen_names=()
	
	# Check if the CSV file exists
	if [ ! -f "$csv_file" ]; then
		echo "Error: devices.csv file not found at $csv_file" >&2
		exit 1
	fi
	
	# Read the CSV file and populate the devices array
	while IFS=',' read -r serial name || [ -n "$serial" ]; do
		# Skip the header row
		if [ $line_number -eq 0 ]; then
			if [ "$serial" != "Serial" ] || [ "$name" != "Device" ]; then
				echo "Warning: Unexpected header in CSV file. Expected 'Serial,Device'" >&2
			fi
		else
			# Check if we've seen this device name before
			if ! [[ " ${seen_names[@]} " =~ " ${name} " ]]; then
				devices+=("$serial|$name")  # Use | as separator to avoid issues with commas in names
				seen_names+=("$name")
			else
				echo "Warning: Ignoring duplicate device name: $name (Serial: $serial)" >&2
			fi
		fi
		((line_number++))
	done < "$csv_file"
	
	if [ ${#devices[@]} -eq 0 ]; then
		echo "Error: No devices found in the CSV file" >&2
		exit 1
	fi
	
	# Sort devices by name
	IFS=$'\n' sorted_devices=($(for d in "${devices[@]}"; do echo "$d"; done | sort -t'|' -k2))
	
	# Output the sorted devices
	for device in "${sorted_devices[@]}"; do
		echo "$device"
	done
}

# Load and sort devices from the CSV file
IFS=$'\n' read -d '' -r -a devices < <(load_devices_from_csv "$devices_csv_path")

# Print the number of devices loaded
echo "Number of devices loaded: ${#devices[@]}"

# Debug output
echo "Loaded devices (sorted by name):"
for device in "${devices[@]}"; do
    IFS='|' read -r serial name <<< "$device"
    echo "  $serial,$name"
done

# Function to generate safe renaming commands for a device
generate_safe_renaming_commands() {
    local device_dir="$1"
    local serial="$2"
    local date="$3"
    local renaming_script="$4"
    local counter=1
    local files_renamed=false
    local phase1_commands=()

    echo "Generating safe renaming commands for $device_dir:"
    echo "# Safe renaming commands for $device_dir" >> "$renaming_script"
    
    # Phase 1: Rename all files to temporary names
    echo "# Phase 1: Rename to temporary names" >> "$renaming_script"
    shopt -s nullglob
    for file in "$device_dir"/*.MP4 "$device_dir"/*.mp4; do
        if [[ -f "$file" ]]; then
            basename=$(basename "$file")
            temp_name="${serial}_${date}_TEMP_$(printf "%05d" $counter).mp4"
            renaming_command="mv \"$file\" \"$device_dir/$temp_name\""
            echo "$renaming_command"
            echo "$renaming_command" >> "$renaming_script"
            phase1_commands+=("$renaming_command")
            if $execute_renaming; then
                eval "$renaming_command"
                files_renamed=true
            fi
            ((counter++))
        fi
    done

    # Phase 2: Rename from temporary names to final names
    echo "" >> "$renaming_script"
    echo "# Phase 2: Rename from temporary to final names" >> "$renaming_script"
    for cmd in "${phase1_commands[@]}"; do
        # Extract the source and destination from the Phase 1 command
        src=$(echo "$cmd" | awk -F'"' '{print $2}')
        dest=$(echo "$cmd" | awk -F'"' '{print $4}')
        
        # Create the final name by removing _TEMP_
        final_name=$(basename "$dest" | sed 's/_TEMP_/_/')
        
        # Construct the Phase 2 command
        phase2_cmd="mv \"$dest\" \"$device_dir/$final_name\""
        echo "$phase2_cmd"
        echo "$phase2_cmd" >> "$renaming_script"
        if $execute_renaming; then
            eval "$phase2_cmd"
            files_renamed=true
        fi
    done

    shopt -u nullglob
    echo "" >> "$renaming_script"  # Add a blank line for readability
    echo "$files_renamed"
}

# Function to generate playlists and check file format for a given date
generate_playlists_and_check_format() {
    local date=$1
    local date_dir="${root_directory}/${date}"
    local concat_playlist="${date_dir}/all_devices_playlist.m3u"
    local format_check_file="${date_dir}/file_format_check.txt"
    local timestamp=$(date "+%Y%m%d_%H%M%S")

    local renaming_script="${date_dir}/file_renaming_${timestamp}.sh"
	local copying_script="${date_dir}/file_copying_to_set_time_${timestamp}.sh"

    echo "Processing date: $date"
    echo "Date directory: $date_dir"
    echo "Concatenated playlist: $concat_playlist"
    echo "Format check file: $format_check_file"
    echo "Renaming script: $renaming_script"
    echo "Copying script: $copying_script"

    # Clear the concatenated playlist and format check file if they exist
    > "$concat_playlist"
    > "$format_check_file"
    echo "#!/bin/bash" > "$renaming_script"
    echo "# Renaming script for $date, generated on $timestamp" >> "$renaming_script"
    echo "" >> "$renaming_script"
	echo "#!/bin/bash" > "$copying_script"
    echo "# Copying script to setup the final time for $date, generated on $timestamp" >> "$copying_script"
    echo "" >> "$copying_script"

    local number_of_videos=0

    echo "Number of devices to process: ${#devices[@]}"
    for device in "${devices[@]}"; do
        IFS='|' read -r serial name <<< "$device"
        echo "Processing device: $name (Serial: $serial)"
        device_dir="${date_dir}/${name}"
        playlist_file="${device_dir}_playlist.m3u"
        
        echo "Device directory: $device_dir"
        echo "Playlist file: $playlist_file"

        # Check if the device directory exists and contains MP4 files
        if [ -d "$device_dir" ]; then
            echo "Directory exists for $name"
            shopt -s nullglob
            mp4_files=( "$device_dir"/*.MP4 "$device_dir"/*.mp4 )
            shopt -u nullglob
            if [ ${#mp4_files[@]} -gt 0 ]; then
                echo "MP4 files found for $name"
                local all_files_valid=true
                local expected_prefix="${serial}_${date}"

                echo "Searching for MP4 files in $device_dir"
                > "$playlist_file"  # Clear the playlist file before populating
                for file in "${mp4_files[@]}"; do
				    ((number_of_videos++))
                    basename=$(basename "$file")
                    echo "Checking file: $basename"
                    if [[ ! $basename =~ ^${expected_prefix} ]]; then
                        echo "Invalid file name format: $basename"
                        all_files_valid=false
                    fi
                    echo "$file" >> "$playlist_file"

					# WRITE THE COPY COMMAND to $copying_script HERE
					copying_command="mv \"$file\" \"$device_dir/${serial}_${date}.mp4\""
            		echo "$copying_command" >> "$copying_script"

                done

                echo "Generated playlist for $name on $date"

                # Append this playlist to the concatenated playlist
                cat "$playlist_file" >> "$concat_playlist"

                # Generate renaming commands if files are not valid
                if [ "$all_files_valid" = false ]; then
                    files_renamed=$(generate_safe_renaming_commands "$device_dir" "$serial" "$date" "$renaming_script")
                    if $execute_renaming && $files_renamed; then
                        # Regenerate playlist after renaming
                        > "$playlist_file"
                        shopt -s nullglob
                        for file in "$device_dir"/*.MP4; do
                            echo "$file" >> "$playlist_file"
                        done
                        shopt -u nullglob
                        # Update concatenated playlist
                        sed -i '' "/$name/d" "$concat_playlist"
                        cat "$playlist_file" >> "$concat_playlist"
                        all_files_valid=true
                    fi
                fi

                # Add result to the format check file
                echo "${name},${all_files_valid}" >> "$format_check_file"
            else
                echo "No MP4 files found for $name"
            fi
        else
            echo "Directory does not exist for $name"
        fi
    done

    # Make the renaming script executable
    chmod +x "$renaming_script"

    echo "Generated concatenated playlist, format check file, and renaming script for all devices on $date"
	echo "Total number of videos found: $number_of_videos"
	echo "Copying script: $copying_script"
    echo "Changing execution mode of copying script:"
	echo "chmod a+x $copying_script"
    chmod a+x $copying_script
    echo "Edit copying script to add times, and execute it by typing:"
    echo "${copying_script}"
}

# Function to validate date format
validate_date() {
    if [[ ! $1 =~ ^[0-9]{8}$ ]]; then
        echo "Invalid date format: $1. Please use YYYYMMDD format."
        return 1
    fi
    return 0
}

# Function to generate dates in a range
generate_date_range() {
    local start_date=$1
    local end_date=$2
    local current_date=$start_date

    while [ $(date -j -f "%Y%m%d" "$current_date" +%s) -le $(date -j -f "%Y%m%d" "$end_date" +%s) ]; do
        echo $current_date
        current_date=$(date -j -v+1d -f "%Y%m%d" "${current_date}" +%Y%m%d)
    done
}

# Parse command line arguments
if [[ "$1" == "-execute_file_renaming" ]]; then
    execute_renaming=true
    shift
fi

if [ "$1" = "-r" ]; then
    if [ $# -ne 3 ]; then
        echo "Usage for date range: $0 [-execute_file_renaming] -r <start_date> <end_date>"
        echo "Dates should be in YYYYMMDD format"
        exit 1
    fi
    start_date=$2
    end_date=$3
    
    if ! validate_date $start_date || ! validate_date $end_date; then
        exit 1
    fi

    dates=$(generate_date_range $start_date $end_date)
else
    if [ $# -eq 0 ]; then
        echo "Usage for individual dates: $0 [-execute_file_renaming] <date1> [date2] [date3] ..."
        echo "Usage for date range: $0 [-execute_file_renaming] -r <start_date> <end_date>"
        echo "Dates should be in YYYYMMDD format"
        exit 1
    fi
    dates="$@"
fi

# Process each date
for date in $dates; do
    if validate_date $date; then
        echo "Processing date: $date"
        generate_playlists_and_check_format "$date"
    fi
done

echo "All playlists generated and file format checks completed."
if $execute_renaming; then
    echo "File renaming has been executed where necessary."
else
    echo "File renaming commands have been generated but not executed. Use -execute_file_renaming to perform renaming:"
    echo "./list_mp4_files.sh  -execute_file_renaming ..."
fi

