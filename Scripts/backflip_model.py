#=============================================================================================
"""
- Retningen skal være i forhold til elementet
- Højden godt over skulderhøjde
- Armene skal være strakte så meget som muligt
- Benene skal være strakte (undtagen i sammenbøjet) og holdes samlede ind til 90° (klokken 3)
- Kroppen skal være strakt, hoftebøjet eller sammenbøjet i forhold til elementet
- Åbningen skal være tydeligt genkendelig

Specifikke fradrag:
- Ingen genkendelig åbning før landing 0,1 - 0,3 (0,1 = 180-150 grader, 0,2 = 150-120 grader, 0,3 under 90 grader)
- Højde i skulderhøjde 0,1
- Højde under skulderhøjde 0,2
"""

"""
1. Start with 0.5 points total (Perfect Score for Backflip)
2. Log Shoulder Height & Hip height at standing position
3. Log Shoulder Height at peak of jump (How do we define the peak? -> Max hip height?) 
4. Determine shoulder classification (above, at, below)
4. Apply deduction (if any) based on shoulder classification
5. Determine knee angle at landing (How do we define landing? -> Max hip height?)
X. Calculate deductions during jump for angle of legs, arms, and back
X. 
"""
#=============================================================================================
# Step 1 - Data Preprocessing
#=============================================================================================
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Smooth joint angles to reduce noise
def smooth_data(data, window_length=15, polyorder=3):
    return savgol_filter(data, window_length, polyorder)

# Example DataFrame with noisy joint angles (sample structure)
# Columns: ['frame', 'time', 'knee_angle', 'hip_angle', 'ankle_angle']
data = pd.DataFrame({
    'frame': np.arange(0, 500),
    'time': np.linspace(0, 5, 500),
    'knee_angle': np.sin(np.linspace(0, 10, 500)) * 90 + 90 + np.random.normal(0, 5, 500),
    'hip_angle': np.sin(np.linspace(0, 10, 500) + 1) * 45 + 45 + np.random.normal(0, 5, 500),
    'ankle_angle': np.cos(np.linspace(0, 10, 500)) * 20 + 20 + np.random.normal(0, 2, 500),
})

# Apply smoothing to joint angles
data['knee_angle_smooth'] = smooth_data(data['knee_angle'])
data['hip_angle_smooth'] = smooth_data(data['hip_angle'])
data['ankle_angle_smooth'] = smooth_data(data['ankle_angle'])


#=============================================================================================
# Step 2 - Identify Key Events
#=============================================================================================
from scipy.signal import find_peaks

# Find peaks (e.g., maximum flexion or extension)
def find_extrema(data, prominence=5):
    peaks, _ = find_peaks(data, prominence=prominence)
    valleys, _ = find_peaks(-data, prominence=prominence)
    return peaks, valleys

# Detect key points for knee and hip angles
knee_peaks, knee_valleys = find_extrema(data['knee_angle_smooth'])
hip_peaks, hip_valleys = find_extrema(data['hip_angle_smooth'])

# Example: Store the detected events
key_events = {
    'knee_peaks': knee_peaks,
    'knee_valleys': knee_valleys,
    'hip_peaks': hip_peaks,
    'hip_valleys': hip_valleys,
}
#=============================================================================================
# Step 3 - Define Computational Measures
#=============================================================================================
# Calculate angular velocity
def calculate_velocity(data, time_col, angle_col):
    return np.gradient(data[angle_col], data[time_col])

def detect_significant_changes(velocity, threshold=10, min_distance=20):
    """
    Detect significant changes in angular velocity with reduced sensitivity.
    - Group nearby changes using min_distance.
    - Select the most prominent change in each group.
    """
    significant_changes = np.where(np.abs(velocity) > threshold)[0]

    # Group nearby changes into clusters and pick the first in each cluster
    clustered_changes = []
    if len(significant_changes) > 0:
        current_cluster = [significant_changes[0]]
        for i in range(1, len(significant_changes)):
            if significant_changes[i] - significant_changes[i - 1] < min_distance:
                current_cluster.append(significant_changes[i])
            else:
                clustered_changes.append(current_cluster[0])  # Take the first in the cluster
                current_cluster = [significant_changes[i]]
        clustered_changes.append(current_cluster[0])  # Add the last cluster
    return np.array(clustered_changes)

# Apply refined hip flexion initiation detection
data['hip_velocity'] = calculate_velocity(data, 'time', 'hip_angle_smooth')
hip_flexion_initiation = detect_significant_changes(data['hip_velocity'], threshold=5, min_distance=30)

print("Refined Hip Flexion Initiation:", hip_flexion_initiation)

#=============================================================================================
# Step 4 - Phase Segmentation
#=============================================================================================
def segment_phases(data, key_events):
    phases = []

    # Phase 1: Start to peak knee flexion
    if len(key_events['knee_peaks']) > 0:
        phases.append({
            'phase': 'Phase 1',
            'start_frame': 0,
            'end_frame': key_events['knee_peaks'][0]
        })

    # Phase 2: Peak knee flexion to knee valley
    if len(key_events['knee_peaks']) > 0 and len(key_events['knee_valleys']) > 0:
        phases.append({
            'phase': 'Phase 2',
            'start_frame': key_events['knee_peaks'][0],
            'end_frame': key_events['knee_valleys'][0]
        })

    # Phase 3: Knee valley to hip flexion initiation
    if len(key_events['knee_valleys']) > 0 and len(hip_flexion_initiation) > 0:
        phases.append({
            'phase': 'Phase 3',
            'start_frame': key_events['knee_valleys'][0],
            'end_frame': hip_flexion_initiation[0]
        })

    # Phase 4: Hip flexion initiation to peak hip flexion
    if len(hip_flexion_initiation) > 0 and len(key_events['hip_peaks']) > 0:
        phases.append({
            'phase': 'Phase 4',
            'start_frame': hip_flexion_initiation[0],
            'end_frame': key_events['hip_peaks'][0]
        })

    # Ensure valid and chronological phases
    phases = [p for p in phases if p['start_frame'] < p['end_frame']]
    phases = sorted(phases, key=lambda x: x['start_frame'])

    return phases

# Segment the phases
phases = segment_phases(data, key_events)

#=============================================================================================
# Step 5 - Derive Scoring Features
#=============================================================================================
# Calculate derived features for scoring
def calculate_scoring_features(data, phases):
    scores = []
    for phase in phases:
        start, end = phase['start_frame'], phase['end_frame']
        
        # Skip invalid or overlapping phases
        if start >= end:
            print(f"Skipping invalid phase: {phase['phase']} (start={start}, end={end})")
            continue
        
        # Extract phase data
        phase_data = data.iloc[start:end]
        
        # Skip empty DataFrames
        if phase_data.empty:
            print(f"Skipping empty phase: {phase['phase']} (start={start}, end={end})")
            continue
        
        # Calculate features (e.g., peak angles, duration)
        peak_knee_angle = phase_data['knee_angle_smooth'].max()
        peak_hip_angle = phase_data['hip_angle_smooth'].max()
        duration = phase_data['time'].iloc[-1] - phase_data['time'].iloc[0]
        
        scores.append({
            'phase': phase['phase'],
            'peak_knee_angle': peak_knee_angle,
            'peak_hip_angle': peak_hip_angle,
            'duration': duration,
        })
    return pd.DataFrame(scores)


# Calculate scoring features for each phase
scoring_features = calculate_scoring_features(data, phases)

#=============================================================================================
# Debugging
#=============================================================================================
print("Detected Key Events:")
print(f"Knee Peaks: {key_events['knee_peaks']}")
print(f"Knee Valleys: {key_events['knee_valleys']}")
print(f"Hip Flexion Initiation: {hip_flexion_initiation}")
print(f"Hip Peaks: {key_events['hip_peaks']}")


# Print segmented phases
for phase in phases:
    print(f"{phase['phase']}: {phase['start_frame']} - {phase['end_frame']}")

#=============================================================================================
# Visualizing Data
#=============================================================================================
# Plot the smoothed data with phase boundaries
import matplotlib.pyplot as plt
plt.plot(data['time'], data['knee_angle_smooth'], label='Knee Angle')
plt.plot(data['time'], data['hip_angle_smooth'], label='Hip Angle')
plt.scatter(data['time'][key_events['knee_peaks']], 
            data['knee_angle_smooth'][key_events['knee_peaks']], 
            color='red', label='Knee Peaks')
plt.scatter(data['time'][key_events['knee_valleys']], 
            data['knee_angle_smooth'][key_events['knee_valleys']], 
            color='blue', label='Knee Valleys')
plt.scatter(data['time'][hip_flexion_initiation], 
            data['hip_angle_smooth'][hip_flexion_initiation], 
            color='green', label='Hip Flexion Initiation')
plt.legend()
plt.show()

#=============================================================================================
