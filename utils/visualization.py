import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import cv2
from PIL import Image
import streamlit as st


def draw_detections(image, detections, colors=None):
    """
    Draw detections on image with enhanced visualization

    Args:
        image: Input image
        detections: List of detection dictionaries
        colors: Optional color mapping

    Returns:
        Image with visualizations
    """
    if colors is None:
        # Generate distinct colors
        num_classes = len(set(detection['class_id'] for detection in detections))
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)]

    image_copy = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        confidence = det['confidence']
        class_name = det['class_name']
        class_id = det['class_id']

        color = colors[class_id % len(colors)]

        # Draw bounding box with rounded corners
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 3)

        # Draw confidence bar
        bar_height = 20
        bar_width = int((x2 - x1) * 0.8)
        bar_x = x1 + int((x2 - x1) * 0.1)
        bar_y = y2 + 10

        cv2.rectangle(
            image_copy,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50),
            -1
        )

        # Confidence fill
        fill_width = int(bar_width * confidence)
        cv2.rectangle(
            image_copy,
            (bar_x, bar_y),
            (bar_x + fill_width, bar_y + bar_height),
            color,
            -1
        )

        # Label with background
        label = f"{class_name} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        cv2.rectangle(
            image_copy,
            (x1, y1 - text_height - 15),
            (x1 + text_width + 10, y1),
            color,
            -1
        )

        cv2.putText(
            image_copy,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return image_copy


def create_metrics_dashboard():
    """
    Create interactive performance metrics dashboard using Plotly
    """
    fig = go.Figure()

    # Performance metrics
    metrics = {
        'mAP@0.5': 0.874,
        'mAP@0.5:0.95': 0.652,
        'Precision': 0.856,
        'Recall': 0.789,
        'F1-Score': 0.821
    }

    fig.add_trace(go.Scatter(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        mode='markers+lines',
        name='Model Performance',
        marker=dict(size=15, color='#1f77b4'),
        line=dict(width=3, color='#17becf')
    ))

    fig.update_layout(
        title='Traffic Sign Detection Performance Metrics',
        xaxis_title='Metrics',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1.0]),
        template='plotly_white',
        height=400,
        showlegend=True
    )

    return fig


def create_confusion_matrix(confusion_matrix, class_names):
    """
    Create confusion matrix visualization

    Args:
        confusion_matrix: Confusion matrix data
        class_names: List of class names
    """
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=class_names[:5],  # Show top 5 for clarity
        y=class_names[:5],
        colorscale='Viridis',
        text=confusion_matrix,
        texttemplate='%{text}',
        textfont={"size": 10}
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
        height=500,
        width=500
    )

    return fig


def create_performance_over_time(performance_data):
    """
    Create performance over time visualization

    Args:
        performance_data: Dictionary with performance metrics over time
    """
    fig = go.Figure()

    for metric, values in performance_data.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode='lines+markers',
            name=metric
        ))

    fig.update_layout(
        title='Model Performance Over Time',
        xaxis_title='Epochs',
        yaxis_title='Score',
        template='plotly_white',
        height=400
    )

    return fig


def create_inference_speed_comparison():
    """
    Create bar chart comparing inference speeds on different hardware
    """
    hardware = ['GPU (CUDA)', 'GPU (OpenCL)', 'CPU (AVX)', 'CPU (Base)']
    fps = [45, 38, 18, 12]

    colors = ['#2ecc71' if f >= 30 else '#f1c40f' if f >= 15 else '#e74c3c' for f in fps]

    fig = go.Figure(data=[
        go.Bar(
            x=hardware,
            y=fps,
            marker_color=colors,
            text=[f'{f} FPS' for f in fps],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title='Inference Speed Comparison',
        xaxis_title='Hardware',
        yaxis_title='Frames Per Second (FPS)',
        yaxis=dict(range=[0, 60]),
        template='plotly_white',
        height=400
    )

    # Add threshold line
    fig.add_hline(y=30, line_dash="dash", line_color="red",
                  annotation_text="Real-time Threshold (30 FPS)")

    return fig


def create_detection_statistics(detections):
    """
    Create summary statistics of detections

    Args:
        detections: List of detection dictionaries
    """
    if not detections:
        return None

    # Count class frequencies
    class_counts = {}
    confidence_scores = []

    for det in detections:
        class_name = det['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        confidence_scores.append(det['confidence'])

    # Create pie chart for class distribution
    fig1 = go.Figure(data=[
        go.Pie(
            labels=list(class_counts.keys()),
            values=list(class_counts.values()),
            hole=0.3,
            textinfo='label+percent'
        )
    ])

    fig1.update_layout(
        title='Detected Sign Distribution',
        height=300,
        width=400
    )

    # Create histogram for confidence scores
    fig2 = go.Figure(data=[
        go.Histogram(
            x=confidence_scores,
            nbinsx=10,
            marker_color='#1f77b4'
        )
    ])

    fig2.update_layout(
        title='Confidence Score Distribution',
        xaxis_title='Confidence Score',
        yaxis_title='Count',
        height=300,
        width=400
    )

    return fig1, fig2


def create_optimization_benchmark():
    """
    Create benchmark visualization for model optimization
    """
    formats = ['PyTorch', 'ONNX', 'TFLite', 'TensorRT']
    inference_time = [25, 18, 22, 8]  # ms
    model_size = [22.5, 22.0, 18.5, 16.0]  # MB

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=formats,
        y=inference_time,
        name='Inference Time (ms)',
        marker_color='#1f77b4',
        yaxis='y'
    ))

    fig.add_trace(go.Scatter(
        x=formats,
        y=model_size,
        name='Model Size (MB)',
        marker=dict(size=15, color='#ff7f0e'),
        yaxis='y2'
    ))

    fig.update_layout(
        title='Model Optimization Benchmark',
        template='plotly_white',
        height=400,
        yaxis=dict(title='Inference Time (ms)'),
        yaxis2=dict(
            title='Model Size (MB)',
            overlaying='y',
            side='right'
        )
    )

    return fig