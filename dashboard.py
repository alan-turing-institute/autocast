"""Streamlit dashboard for comparing experiment runs.

This dashboard provides three main features:
1. Metrics Comparison - Compare metrics across multiple runs with bar charts
2. Image Comparison - View coverage plots side-by-side from two runs
3. Video Comparison - Compare rollout videos side-by-side from two runs
"""

import base64
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


@st.cache_data
def discover_runs(base_dir: Path) -> list[str]:
    """Discover all run directories containing eval/ subdirectories.

    Parameters
    ----------
    base_dir : Path
        Base directory to search for runs (e.g., outputs/)

    Returns
    -------
    list[str]
        List of run paths relative to base_dir
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    runs = []
    for eval_dir in base_path.rglob("eval"):
        if eval_dir.is_dir():
            # Get the parent directory (the run directory)
            run_dir = eval_dir.parent
            # Get path relative to base_dir
            rel_path = run_dir.relative_to(base_path)
            runs.append(str(rel_path))

    return sorted(runs)


@st.cache_data
def load_metrics(base_dir: Path, run_path: str, metric_type: str) -> pd.DataFrame:
    """Load metrics CSV file for a given run.

    Parameters
    ----------
    base_dir : Path
        Base directory containing runs
    run_path : str
        Relative path to run directory
    metric_type : str
        Type of metrics: "Evaluation" or "Rollout"

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics, or empty DataFrame if file not found
    """
    run_full_path = Path(base_dir) / run_path
    csv_name = (
        "evaluation_metrics.csv" if metric_type == "Evaluation" else "rollout_metrics.csv"
    )
    csv_path = run_full_path / "eval" / csv_name

    if not csv_path.exists():
        return pd.DataFrame()

    return pd.read_csv(csv_path)


@st.cache_data
def get_available_images(base_dir: Path, run_path: str) -> dict[str, Path]:
    """Get available PNG images for a run.

    Parameters
    ----------
    base_dir : Path
        Base directory containing runs
    run_path : str
        Relative path to run directory

    Returns
    -------
    dict[str, Path]
        Mapping of image names to file paths
    """
    eval_dir = Path(base_dir) / run_path / "eval"
    if not eval_dir.exists():
        return {}

    images = {}
    for png_file in eval_dir.glob("*.png"):
        # Use stem (filename without extension) as the key
        images[png_file.stem] = png_file

    return images


@st.cache_data
def get_available_videos(base_dir: Path, run_path: str) -> dict[str, Path]:
    """Get available MP4 videos for a run.

    Parameters
    ----------
    base_dir : Path
        Base directory containing runs
    run_path : str
        Relative path to run directory

    Returns
    -------
    dict[str, Path]
        Mapping of video names to file paths
    """
    video_dir = Path(base_dir) / run_path / "eval" / "videos"
    if not video_dir.exists():
        return {}

    videos = {}
    for mp4_file in video_dir.glob("*.mp4"):
        # Use stem (filename without extension) as the key
        videos[mp4_file.stem] = mp4_file

    return videos


def create_synchronized_video_player(
    video1_path: Path, video2_path: Path, title1: str, title2: str
) -> str:
    """Create HTML for synchronized video players.

    Parameters
    ----------
    video1_path : Path
        Path to first video file
    video2_path : Path
        Path to second video file
    title1 : str
        Title for first video
    title2 : str
        Title for second video

    Returns
    -------
    str
        HTML string with synchronized video players
    """
    # Read video files and encode as base64
    with open(video1_path, "rb") as f:
        video1_data = base64.b64encode(f.read()).decode()

    with open(video2_path, "rb") as f:
        video2_data = base64.b64encode(f.read()).decode()

    html = f"""
    <style>
        .video-container {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .video-wrapper {{
            flex: 1;
            text-align: center;
        }}
        .video-title {{
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 14px;
            color: #262730;
        }}
        video {{
            width: 100%;
            height: auto;
            background: #000;
        }}
        .controls {{
            margin-top: 20px;
            text-align: center;
        }}
        .control-btn {{
            margin: 5px;
            padding: 10px 20px;
            font-size: 14px;
            cursor: pointer;
            background-color: #FF4B4B;
            color: white;
            border: none;
            border-radius: 4px;
        }}
        .control-btn:hover {{
            background-color: #FF6B6B;
        }}
        .time-display {{
            margin-top: 10px;
            font-size: 14px;
            color: #262730;
        }}
    </style>

    <div class="video-container">
        <div class="video-wrapper">
            <div class="video-title">{title1}</div>
            <video id="video1" controls>
                <source src="data:video/mp4;base64,{video1_data}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        <div class="video-wrapper">
            <div class="video-title">{title2}</div>
            <video id="video2" controls>
                <source src="data:video/mp4;base64,{video2_data}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
    </div>

    <div class="controls">
        <button class="control-btn" onclick="playPause()">⏯ Play/Pause</button>
        <button class="control-btn" onclick="restart()">⏮ Restart</button>
        <button class="control-btn" onclick="skipBackward()">⏪ -5s</button>
        <button class="control-btn" onclick="skipForward()">⏩ +5s</button>
        <button class="control-btn" onclick="changeSpeed(0.5)">0.5x</button>
        <button class="control-btn" onclick="changeSpeed(1.0)">1.0x</button>
        <button class="control-btn" onclick="changeSpeed(2.0)">2.0x</button>
        <div class="time-display" id="timeDisplay">0:00 / 0:00</div>
    </div>

    <script>
        const video1 = document.getElementById('video1');
        const video2 = document.getElementById('video2');
        const timeDisplay = document.getElementById('timeDisplay');

        // Synchronize video2 with video1
        function syncVideos(master, slave) {{
            // Sync play/pause
            master.addEventListener('play', () => slave.play());
            master.addEventListener('pause', () => slave.pause());

            // Sync seeking
            master.addEventListener('seeked', () => {{
                slave.currentTime = master.currentTime;
            }});

            // Sync playback rate
            master.addEventListener('ratechange', () => {{
                slave.playbackRate = master.playbackRate;
            }});

            // Update time display
            master.addEventListener('timeupdate', () => {{
                const current = formatTime(master.currentTime);
                const duration = formatTime(master.duration);
                timeDisplay.textContent = `${{current}} / ${{duration}}`;
            }});
        }}

        // Set up bidirectional sync
        syncVideos(video1, video2);
        syncVideos(video2, video1);

        function playPause() {{
            if (video1.paused) {{
                video1.play();
            }} else {{
                video1.pause();
            }}
        }}

        function restart() {{
            video1.currentTime = 0;
            video2.currentTime = 0;
        }}

        function skipBackward() {{
            video1.currentTime = Math.max(0, video1.currentTime - 5);
            video2.currentTime = video1.currentTime;
        }}

        function skipForward() {{
            video1.currentTime = Math.min(video1.duration, video1.currentTime + 5);
            video2.currentTime = video1.currentTime;
        }}

        function changeSpeed(rate) {{
            video1.playbackRate = rate;
            video2.playbackRate = rate;
        }}

        function formatTime(seconds) {{
            if (isNaN(seconds)) return '0:00';
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${{mins}}:${{secs.toString().padStart(2, '0')}}`;
        }}
    </script>
    """
    return html


def plot_metric_comparison(
    base_dir: Path,
    selected_runs: list[str],
    metric: str,
    metric_type: str,
    window: str | None = None,
    batch_idx: str = "all",
) -> go.Figure:
    """Create bar chart comparing a metric across runs.

    Parameters
    ----------
    base_dir : Path
        Base directory containing runs
    selected_runs : list[str]
        List of run paths to compare
    metric : str
        Metric to plot (mse, mae, rmse, vrmse, coverage)
    metric_type : str
        Type of metrics: "Evaluation" or "Rollout"
    window : str | None
        Time window for rollout metrics (e.g., "0-1", "6-12")
    batch_idx : str
        Batch index to filter ("all" for aggregate)

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    data = []

    for run in selected_runs:
        df = load_metrics(base_dir, run, metric_type)
        if df.empty:
            continue

        # Filter by window if specified
        if window is not None:
            df = df[df["window"] == window]
        else:
            # For evaluation metrics, use "all" window
            df = df[df["window"] == "all"]

        # Filter by batch_idx
        df = df[df["batch_idx"] == batch_idx]

        if not df.empty and metric in df.columns:
            metric_value = df[metric].iloc[0]
            data.append({"Run": run, metric.upper(): metric_value})

    if not data:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected runs/metric",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Create DataFrame and plot
    plot_df = pd.DataFrame(data)
    fig = px.bar(
        plot_df,
        y="Run",
        x=metric.upper(),
        orientation="h",
        title=f"{metric.upper()} Comparison",
    )
    fig.update_layout(height=max(300, len(data) * 50))

    return fig


def create_leaderboard(
    base_dir: Path,
    runs: list[str],
    metric_type: str,
    window: str | None = None,
    batch_idx: str = "all",
) -> pd.DataFrame:
    """Create leaderboard DataFrame ranking runs by all metrics.

    Parameters
    ----------
    base_dir : Path
        Base directory containing runs
    runs : list[str]
        List of all run paths
    metric_type : str
        Type of metrics: "Evaluation" or "Rollout"
    window : str | None
        Time window for rollout metrics (e.g., "0-1", "6-12")
    batch_idx : str
        Batch index to filter ("all" for aggregate)

    Returns
    -------
    pd.DataFrame
        DataFrame with runs and their metrics, sorted by rank
    """
    data = []
    metrics = ["mse", "mae", "rmse", "vrmse", "coverage"]

    for run in runs:
        df = load_metrics(base_dir, run, metric_type)
        if df.empty:
            continue

        # Filter by window if specified
        if window is not None:
            df = df[df["window"] == window]
        else:
            # For evaluation metrics, use "all" window
            df = df[df["window"] == "all"]

        # Filter by batch_idx
        df = df[df["batch_idx"] == batch_idx]

        if not df.empty:
            row = {"Run": run}
            for metric in metrics:
                if metric in df.columns:
                    row[metric.upper()] = df[metric].iloc[0]
            data.append(row)

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def plot_leaderboard_heatmap(
    base_dir: Path,
    runs: list[str],
    batch_idx: str = "all",
) -> go.Figure:
    """Create heatmap showing metric rankings for all runs across all windows.

    Parameters
    ----------
    base_dir : Path
        Base directory containing runs
    runs : list[str]
        List of all run paths
    batch_idx : str
        Batch index to filter ("all" for aggregate)

    Returns
    -------
    go.Figure
        Plotly heatmap figure
    """
    # Collect all metrics from evaluation and all rollout windows
    base_metrics = ["mse", "mae", "rmse", "vrmse", "coverage"]
    windows = [None, "0-1", "6-12", "13-30", "31-99"]  # None = Evaluation

    all_data = []

    for run in runs:
        row_data = {"Run": run}

        # Get evaluation metrics
        eval_df = load_metrics(base_dir, run, "Evaluation")
        if not eval_df.empty:
            eval_filtered = eval_df[(eval_df["window"] == "all") & (eval_df["batch_idx"] == batch_idx)]
            if not eval_filtered.empty:
                for metric in base_metrics:
                    if metric in eval_filtered.columns:
                        row_data[f"{metric.upper()} (Eval)"] = eval_filtered[metric].iloc[0]

        # Get rollout metrics for all windows
        rollout_df = load_metrics(base_dir, run, "Rollout")
        if not rollout_df.empty:
            for window in windows[1:]:  # Skip None (already did eval)
                window_filtered = rollout_df[(rollout_df["window"] == window) & (rollout_df["batch_idx"] == batch_idx)]
                if not window_filtered.empty:
                    for metric in base_metrics:
                        if metric in window_filtered.columns:
                            row_data[f"{metric.upper()} ({window})"] = window_filtered[metric].iloc[0]

        all_data.append(row_data)

    if not all_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Get all metric columns (everything except "Run")
    metric_cols = [col for col in df.columns if col != "Run"]

    # Create rank matrix (color by rank for each metric independently)
    rank_matrix = []
    value_matrix = []  # For hover text

    for _, row in df.iterrows():
        rank_row = []
        value_row = []
        for metric_col in metric_cols:
            if metric_col in row and pd.notna(row[metric_col]):
                value = row[metric_col]
                value_row.append(value)
                # All metrics: lower is better (ascending rank)
                rank = (df[metric_col] <= value).sum()
                rank_row.append(rank)
            else:
                value_row.append(None)
                rank_row.append(None)
        rank_matrix.append(rank_row)
        value_matrix.append(value_row)

    # Create hover text with both rank and value
    hover_text = []
    for i, run in enumerate(df["Run"]):
        hover_row = []
        for j, metric_col in enumerate(metric_cols):
            rank = rank_matrix[i][j]
            value = value_matrix[i][j]
            if value is not None and rank is not None:
                hover_row.append(
                    f"Run: {run}<br>"
                    f"Metric: {metric_col}<br>"
                    f"Rank: {rank}<br>"
                    f"Value: {value:.6f}"
                )
            else:
                hover_row.append(f"Run: {run}<br>Metric: {metric_col}<br>No data")
        hover_text.append(hover_row)

    # Create heatmap with single color gradient (dark blue = best/rank 1, light = worst)
    fig = go.Figure(data=go.Heatmap(
        z=rank_matrix,
        x=metric_cols,
        y=df["Run"],
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorscale='Blues_r',  # Reversed Blues: dark blue = best (low rank), light = worst
        colorbar=dict(title="Rank"),
        reversescale=False,
    ))

    fig.update_layout(
        title="Model Leaderboard - All Metrics (Dark Blue=Best, Light=Worst, Lower is Better for All)",
        xaxis_title="Metric",
        yaxis_title="Run",
        height=max(400, len(df) * 40),
        xaxis=dict(tickangle=-45),
    )

    return fig


def plot_metric_crossplot(
    base_dir: Path,
    selected_runs: list[str],
    metric_x: str,
    metric_y: str,
    metric_type: str,
    window: str | None = None,
    batch_idx: str = "all",
) -> go.Figure:
    """Create scatter plot comparing two metrics across runs.

    Parameters
    ----------
    base_dir : Path
        Base directory containing runs
    selected_runs : list[str]
        List of run paths to compare
    metric_x : str
        Metric for X-axis (mse, mae, rmse, vrmse, coverage)
    metric_y : str
        Metric for Y-axis (mse, mae, rmse, vrmse, coverage)
    metric_type : str
        Type of metrics: "Evaluation" or "Rollout"
    window : str | None
        Time window for rollout metrics (e.g., "0-1", "6-12")
    batch_idx : str
        Batch index to filter ("all" for aggregate)

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    data = []

    for run in selected_runs:
        df = load_metrics(base_dir, run, metric_type)
        if df.empty:
            continue

        # Filter by window if specified
        if window is not None:
            df = df[df["window"] == window]
        else:
            # For evaluation metrics, use "all" window
            df = df[df["window"] == "all"]

        # Filter by batch_idx
        df = df[df["batch_idx"] == batch_idx]

        if not df.empty and metric_x in df.columns and metric_y in df.columns:
            x_value = df[metric_x].iloc[0]
            y_value = df[metric_y].iloc[0]
            data.append({
                "Run": run,
                metric_x.upper(): x_value,
                metric_y.upper(): y_value,
            })

    if not data:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected runs/metrics",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Create DataFrame and plot
    plot_df = pd.DataFrame(data)
    fig = px.scatter(
        plot_df,
        x=metric_x.upper(),
        y=metric_y.upper(),
        color="Run",
        hover_name="Run",
        title=f"{metric_y.upper()} vs {metric_x.upper()}",
    )
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(height=600, showlegend=True)

    return fig


# Streamlit UI
st.set_page_config(page_title="Autocast Dashboard", layout="wide")
st.title("Autocast Experiment Dashboard")

# Sidebar for global settings
with st.sidebar:
    st.header("Settings")
    base_dir_str = st.text_input("Output Directory", "outputs/")
    base_dir = Path(base_dir_str)

    if not base_dir.exists():
        st.error(f"Directory not found: {base_dir_str}")
        st.stop()

    runs = discover_runs(base_dir)

    if not runs:
        st.error(f"No runs found in {base_dir_str}")
        st.info("Looking for directories with 'eval/' subdirectories")
        st.stop()

    st.success(f"Found {len(runs)} runs")

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Metrics Comparison", "🖼️ Image Comparison", "🎥 Video Comparison", "🏆 Leaderboard"]
)

# Tab 1: Metrics Comparison
with tab1:
    st.header("Metrics Comparison")

    # Visualization mode selector
    viz_mode = st.radio(
        "Visualization Mode",
        ["Bar Chart", "Cross Plot"],
        horizontal=True,
        help="Bar Chart: Compare a single metric across runs. Cross Plot: Plot two metrics against each other."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        selected_runs = st.multiselect(
            "Select Runs",
            runs,
            default=runs,  # Select all runs by default
            help="Select one or more runs to compare",
        )

        metric_type = st.selectbox("Metric Type", ["Evaluation", "Rollout"])

    with col2:
        if viz_mode == "Bar Chart":
            metric = st.selectbox(
                "Metric", ["mse", "mae", "rmse", "vrmse", "coverage"], index=4
            )
        else:  # Cross Plot
            metric_x = st.selectbox(
                "X-Axis Metric", ["mse", "mae", "rmse", "vrmse", "coverage"], index=0, key="metric_x"
            )
            metric_y = st.selectbox(
                "Y-Axis Metric", ["mse", "mae", "rmse", "vrmse", "coverage"], index=4, key="metric_y"
            )

        if metric_type == "Rollout":
            window = st.selectbox("Time Window", ["0-1", "6-12", "13-30", "31-99"])
        else:
            window = None

    batch_idx = st.selectbox(
        "Batch",
        ["all"],
        help="Select batch index (currently showing aggregated 'all' only)",
    )

    if selected_runs:
        if viz_mode == "Bar Chart":
            fig = plot_metric_comparison(
                base_dir, selected_runs, metric, metric_type, window, batch_idx
            )
        else:  # Cross Plot
            fig = plot_metric_crossplot(
                base_dir, selected_runs, metric_x, metric_y, metric_type, window, batch_idx
            )
        st.plotly_chart(fig, width='stretch')

        # Show complete metrics table underneath
        st.markdown("---")
        st.subheader("Complete Metrics Table")
        metrics_table_df = create_leaderboard(base_dir, selected_runs, metric_type, window, batch_idx)
        if not metrics_table_df.empty:
            st.dataframe(metrics_table_df, use_container_width=True, hide_index=True)
        else:
            st.info("No metrics data available for selected runs")
    else:
        st.info("Select one or more runs to display metrics comparison")

# Tab 2: Image Comparison
with tab2:
    st.header("Image Comparison")

    # Initialize session state for image lock
    if "image_lock" not in st.session_state:
        st.session_state.image_lock = True

    # Lock button at the top left
    col_lock, col_spacer = st.columns([1, 9])
    with col_lock:
        lock_icon = "🔒" if st.session_state.image_lock else "🔓"
        lock_label = "Locked" if st.session_state.image_lock else "Unlocked"
        if st.button(f"{lock_icon} {lock_label}", key="img_lock_btn"):
            st.session_state.image_lock = not st.session_state.image_lock
            st.rerun()

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Run 1")
        run1 = st.selectbox("Select Run 1", runs, key="run1_img")

        if run1:
            images1 = get_available_images(base_dir, run1)

            if images1:
                if st.session_state.image_lock:
                    # When locked, use a shared selection
                    img_name = st.selectbox(
                        "Image Type (Both)", list(images1.keys()), key="img_shared"
                    )
                    img1_name = img_name
                else:
                    # When unlocked, use separate selections
                    img1_name = st.selectbox(
                        "Image Type", list(images1.keys()), key="img1"
                    )

                st.image(
                    str(images1[img1_name]),
                    caption=f"{run1} - {img1_name}",
                    width='stretch',
                )
            else:
                st.warning(f"No images found for {run1}")

    with col2:
        st.subheader("Run 2")
        run2 = st.selectbox("Select Run 2", runs, key="run2_img")

        if run2:
            images2 = get_available_images(base_dir, run2)

            if images2:
                if st.session_state.image_lock:
                    # When locked, use the same selection as Run 1
                    if "img_shared" in st.session_state:
                        img2_name = st.session_state.img_shared
                    else:
                        img2_name = list(images2.keys())[0]
                    st.info(f"Locked to: {img2_name}")
                else:
                    # When unlocked, use separate selection
                    img2_name = st.selectbox(
                        "Image Type", list(images2.keys()), key="img2"
                    )

                if img2_name in images2:
                    st.image(
                        str(images2[img2_name]),
                        caption=f"{run2} - {img2_name}",
                        width='stretch',
                    )
                else:
                    st.warning(f"Image '{img2_name}' not found in {run2}")
            else:
                st.warning(f"No images found for {run2}")

# Tab 3: Video Comparison
with tab3:
    st.header("Video Comparison")

    # Shared controls at the top
    col1, col2, col3 = st.columns(3)

    with col1:
        run1_vid = st.selectbox("Select Run 1", runs, key="run1_vid")

    with col2:
        run2_vid = st.selectbox("Select Run 2", runs, key="run2_vid")

    # Get available videos from both runs
    if run1_vid and run2_vid:
        videos1 = get_available_videos(base_dir, run1_vid)
        videos2 = get_available_videos(base_dir, run2_vid)

        # Find common videos available in both runs
        common_videos = set(videos1.keys()) & set(videos2.keys())

        if common_videos:
            with col3:
                selected_video = st.selectbox(
                    "Select Video",
                    sorted(list(common_videos)),
                    key="selected_vid"
                )

            # Display synchronized videos
            st.markdown("---")
            st.info("💡 Use the controls below to play/pause, seek, and change playback speed for both videos simultaneously")

            video1_path = videos1[selected_video]
            video2_path = videos2[selected_video]

            html_player = create_synchronized_video_player(
                video1_path,
                video2_path,
                f"Run 1: {run1_vid}",
                f"Run 2: {run2_vid}"
            )

            components.html(html_player, height=800, scrolling=True)
        else:
            st.warning("No common videos found between the selected runs")
            if not videos1:
                st.info(f"No videos found for Run 1: {run1_vid}")
            if not videos2:
                st.info(f"No videos found for Run 2: {run2_vid}")
    else:
        st.info("Select both runs to compare videos")

# Tab 4: Leaderboard
with tab4:
    st.header("Model Leaderboard")
    st.info("📊 Heatmap showing all metrics (Evaluation + All Rollout Windows). Colored by rank for each metric independently. Dark Blue = Best, Light = Worst. Lower is better for ALL metrics.")

    batch_idx_lb = st.selectbox(
        "Batch",
        ["all"],
        help="Select batch index (currently showing aggregated 'all' only)",
        key="lb_batch"
    )

    st.markdown("---")

    # Create and display heatmap
    fig = plot_leaderboard_heatmap(base_dir, runs, batch_idx_lb)
    st.plotly_chart(fig, use_container_width=True)
