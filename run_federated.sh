#!/bin/bash

NUM_CLIENTS=11
OUTPUT_DIR="Outputs"

# 1) Remove and recreate the directory to ensure it's empty
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# 2) Start the server and redirect its output
python server.py > "$OUTPUT_DIR/Server.log" 2>&1 &
sleep 2  # Wait for the server to start

# 3) Start clients and redirect output
for ((i=0; i<NUM_CLIENTS; i++))
do
    python client.py $i > "$OUTPUT_DIR/Client$i.log" 2>&1 &

done

wait  # Wait for all background processes to finish