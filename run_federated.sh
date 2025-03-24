#!/bin/bash

NUM_CLIENTS=6
OUTPUT_DIR="./Outputs"
TRAIN_DIR="./Dataset/Train"
TEST_DIR="./Dataset/Test"
VISUAL_DIR="./Visualizations"
MODEL_DIR="./models"

# 1) Remove and recreate the directory to ensure it's empty
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# 2) Remove and recreate the data preparation side directories to ensure it's empty
rm -rf "$TRAIN_DIR"
mkdir -p "$TRAIN_DIR"

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

rm -rf "$VISUAL_DIR"
mkdir -p "$VISUAL_DIR"

# 3) Remove and recreate the model directories to ensure it's empty

rm -rf "$MODEL_DIR"
mkdir -p "$MODEL_DIR"

for ((i=0; i<NUM_CLIENTS; i++))
do
    rm -rf "$MODEL_DIR/Client$i"
    mkdir -p "$MODEL_DIR/Client$i"
done

# 4) Running the data preparation script to prepare the data for the clients and visualize it
python data_prep.py > "$OUTPUT_DIR/DataPrep.log" 2>&1

# 5) Start the server and redirect its output
python server.py > "$OUTPUT_DIR/Server.log" 2>&1 &
sleep 2  # Wait for the server to start

# 6) Start clients and redirect output
for ((i=0; i<NUM_CLIENTS; i++))
do
    python client.py $i > "$OUTPUT_DIR/Client$i.log" 2>&1 &

done

wait  # Wait for all background processes to finish
