
import matplotlib.pyplot as plt
import numpy as np

def plot_joint_angle(df, joint_name, fig_size = (14, 8)):
    """
    Plots the joint angle over time for multiple participants from a combined DataFrame,
    distinguishing between 'good' and 'bad' participants by line style and between video numbers.
    
    Parameters:
    df (pd.DataFrame): A DataFrame containing joint angle data with columns 'time', 'participant_id',
                       'evaluation', and 'video_number'.
    joint_name (str): The name of the joint angle column to plot (e.g., 'right_knee', 'left_knee').
    """
    plt.figure(figsize=fig_size)
    
    # Generate a color map for unique participants
    unique_participants = df['participant_id'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_participants)))
    color_map = {participant: color for participant, color in zip(unique_participants, colors)}
    
    # Plot the joint angle for each participant, differentiating by video number
    for participant_id in unique_participants:
        participant_data = df[df['participant_id'] == participant_id]
        
        if joint_name in participant_data.columns:
            unique_videos = participant_data['video_number'].unique()
            
            for video_number in unique_videos:
                video_data = participant_data[participant_data['video_number'] == video_number]
                evaluation = video_data['evaluation'].iloc[0]  # Get the evaluation value for the participant
                line_style = '-' if evaluation == 'good' else '--'
                
                # Plot the joint angle for the participant and video
                plt.plot(video_data['time'], video_data[joint_name],
                         label=f'{evaluation} - {participant_id} (Video {video_number})',
                         color=color_map[participant_id], linestyle=line_style)
        else:
            print(f"Warning: '{joint_name}' not found for {participant_id}")
    
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.title(f'{joint_name.replace("_", " ").title()} Angle Over Time for Multiple Participants and Videos')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_all_joints(df, joint_names, grid=True, fig_size = (14, 8)):
    """
    Plots all specified joint angles over time in a single plot for each participant,
    differentiating between video numbers using line styles. Optionally arranges the plots in a grid.
    
    Parameters:
    df (pd.DataFrame): A DataFrame containing joint angle data with columns 'time', 'participant_id',
                       'evaluation', 'video_number', and the joint columns specified in joint_names.
    joint_names (list of str): A list of joint angle column names to plot (e.g., ['right_knee', 'left_knee']).
    grid (bool): If True, arranges the plots in a grid. Otherwise, plots one at a time.
    """
    unique_participants = df['participant_id'].unique()
    line_styles = ['-', '--', '-.', ':']  # List of line styles to differentiate videos

    if grid:
        num_participants = len(unique_participants)
        num_cols = 2  # Number of columns in the grid
        num_rows = int(np.ceil(num_participants / num_cols))  # Calculate the number of rows
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_size[0],fig_size[1] * num_rows))
        axes = axes.flatten()  # Flatten to easily index if there are multiple subplots

    for idx, participant_id in enumerate(unique_participants):
        # Filter data for the current participant
        participant_data = df[df['participant_id'] == participant_id]
        evaluation = participant_data['evaluation'].iloc[0]  # Get the evaluation type
        
        if grid:
            ax = axes[idx]
        else:
            plt.figure(figsize=fig_size)
            ax = plt.gca()  # Get current axis
        
        # Plot all specified joint angles for the participant, differentiating by video number
        unique_videos = participant_data['video_number'].unique()
        for i, video_number in enumerate(unique_videos):
            video_data = participant_data[participant_data['video_number'] == video_number]
            current_line_style = line_styles[i % len(line_styles)]  # Cycle through line styles
            
            for joint_name in joint_names:
                if joint_name in video_data.columns:
                    ax.plot(video_data['time'], video_data[joint_name],
                            label=f'Video {video_number} - {joint_name.replace("_", " ").title()}',
                            linestyle=current_line_style)
                else:
                    print(f"Warning: '{joint_name}' not found for {participant_id} in Video {video_number}")
        
        # Set plot details
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f'Joint Angles Over Time for {participant_id} ({evaluation})')
        ax.legend()
        ax.grid(True)
        
        if not grid:
            plt.show()

    if grid:
        # Hide any unused subplots
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()



def plot_hip_trajectory(df):
    """
    Plots the hip position (X and Y) for multiple participants from a combined DataFrame,
    distinguishing between 'good' and 'bad' participants by line style.
    
    Parameters:
    df (pd.DataFrame): A DataFrame containing motion capture data with columns 'Hip_X', 'Hip_Y',
                       'participant_id', and 'evaluation'.
    """
    plt.figure(figsize=(12, 6))
    
    # Generate a color map for unique participants
    unique_participants = df['participant_id'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_participants)))
    color_map = {participant: color for participant, color in zip(unique_participants, colors)}
    
    # Plot hip trajectory for each participant
    for participant_id in unique_participants:
        participant_data = df[df['participant_id'] == participant_id]
        if 'Hip_X' in participant_data.columns and 'Hip_Y' in participant_data.columns:
            evaluation = participant_data['evaluation'].iloc[0]  # Get the evaluation value for the participant
            line_style = '-' if evaluation == 'good' else '--'
            plt.plot(participant_data['Hip_X'], participant_data['Hip_Y'],
                     label=f'{evaluation} - {participant_id}',
                     color=color_map[participant_id], linestyle=line_style)
        else:
            print(f"Warning: 'Hip_X' or 'Hip_Y' not found for {participant_id}")
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Hip Position in X-Y Plane Over Time for Multiple Participants')
    plt.legend()
    plt.grid(True)
    plt.show()
