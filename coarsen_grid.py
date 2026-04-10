import numpy as np
import time

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "LA3_d0_v1_uint8_unnormalised_338_338_283.raw"
OUTPUT_FILE = "LA3_coarse_169_169_142.raw"

# Original Dimensions: (Z, Y, X) matching the C++ memory layout
ORIGINAL_SHAPE = (283, 338, 338) 

# THESE MUST MATCH THE C++ RAWType ENUMERATION!
VAL_AIR = 1
VAL_PORE = 2
VAL_ROCK = 3
VAL_SULPHIDE = 4
VAL_PRECIPITATE = 5

def main():
    print(f"Loading high-resolution grid: {INPUT_FILE}...")
    t0 = time.time()
    
    # 1. Load the RAW file
    data = np.fromfile(INPUT_FILE, dtype=np.uint8).reshape(ORIGINAL_SHAPE)
    
    # 2. Pad the Z-axis (283 -> 284) with Air so it divides perfectly by 2
    print("Padding Z-axis to make dimensions even...")
    padded_data = np.full((ORIGINAL_SHAPE[0] + 1, ORIGINAL_SHAPE[1], ORIGINAL_SHAPE[2]), 
                          fill_value=VAL_AIR, dtype=np.uint8)
    padded_data[:-1, :, :] = data
    
    new_z, new_y, new_x = padded_data.shape
    print(f"Padded shape: ({new_z}, {new_y}, {new_x})")
    
    # 3. Vectorized Reshaping to group into 2x2x2 blocks
    # We reshape to (Z/2, 2, Y/2, 2, X/2, 2) then transpose to group the block elements together
    print("Reshaping into 2x2x2 physical blocks...")
    blocks = padded_data.reshape(new_z // 2, 2, new_y // 2, 2, new_x // 2, 2)
    blocks = blocks.transpose(0, 2, 4, 1, 3, 5)
    blocks = blocks.reshape(new_z // 2, new_y // 2, new_x // 2, 8)
    
    # 4. Apply Ranked Priority Voting
    print("Applying Ranked Priority Voting...")
    # Base case: Default everything to Rock (Priority 4)
    coarse_grid = np.full((new_z // 2, new_y // 2, new_x // 2), VAL_ROCK, dtype=np.uint8)
    
    # Priority 3: Pore overwrites Rock
    coarse_grid[np.any(blocks == VAL_PORE, axis=-1)] = VAL_PORE
    
    # Priority 2: Air overwrites Pore and Rock (Preserves the boundary envelope)
    coarse_grid[np.any(blocks == VAL_AIR, axis=-1)] = VAL_AIR
    
    # Priority 1: Sulphide overwrites EVERYTHING (Preserves trace reactive sites)
    coarse_grid[np.any(blocks == VAL_SULPHIDE, axis=-1)] = VAL_SULPHIDE

    # 5. Save the Coarsened Grid
    out_shape = coarse_grid.shape
    print(f"Upscaling complete. New shape: {out_shape}")
    print(f"Saving optimized geometry to {OUTPUT_FILE}...")
    coarse_grid.tofile(OUTPUT_FILE)
    
    t1 = time.time()
    print(f"Script finished in {t1 - t0:.2f} seconds.")

if __name__ == "__main__":
    main()