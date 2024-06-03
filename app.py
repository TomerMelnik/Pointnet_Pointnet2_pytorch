import streamlit as st
import numpy as np
import os
import importlib
import plotly.express as px
import torch
from tqdm import tqdm
from test_partseg import seg_classes, to_categorical
from data_utils.ShapeNetDataLoader import PartNormalDataset
from download_data import download_and_extract_data, DATA_URL


@st.cache_data()
def load_dataset(split="test", npoints=2500, normal_channel=False):
    download_and_extract_data(url=DATA_URL)

    return PartNormalDataset(
        root="./data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
        npoints=npoints,
        split=split,
        normal_channel=normal_channel,
    )


@st.cache_resource()
def load_model(experiment_dir, num_part):
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    
    checkpoint = torch.load(os.path.join(experiment_dir, 'checkpoints', 'best_model.pth'), map_location=torch.device('cpu'))
    
    normal_channel = checkpoint.get('normal_channel', True)
    classifier = MODEL.get_model(num_part, normal_channel=normal_channel)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    
    return classifier.eval()



@st.cache_data()
def get_class_samples(_dataset, class_choice, num_samples):
    indices = [i for i, (cls, _) in enumerate(_dataset.datapath) if cls == class_choice]
    selected_indices = np.random.choice(indices, num_samples, replace=False)
    samples = [_dataset[i] for i in selected_indices]
    return samples

@st.cache_data()
def get_segmentation_results(_samples, normal_channel=True, num_votes=3):
    device = torch.device('cpu')
    num_classes = 16
    num_part = 50
    classifier = load_model('log/part_seg/pointnet2_part_seg_msg', num_part)

    seg_label_to_cat = {label: cat for cat, labels in seg_classes.items() for label in labels}
    
    classifier = classifier.to(device).eval()
    segmentation_results = []

    for points, label, target in _samples:
        # Convert to tensors and move to device
        points = torch.tensor(points).float().to(device)
        label = torch.tensor([label]).long().to(device)  # Ensure label is a 1D tensor
        target = torch.tensor(target).long().to(device)

        # Debugging: Print shapes
        print(f"points shape: {points.shape}, label shape: {label.shape}, target shape: {target.shape}")

        # Ensure points tensor has the correct shape
        if points.ndimension() == 2:
            points = points.unsqueeze(0)  # Add batch dimension if missing
        if points.ndimension() != 3 or points.shape[2] not in [3, 6]:
            raise ValueError(f"Unexpected points shape: {points.shape}")

        # Ensure target tensor has the correct shape
        if target.ndimension() == 1:
            target = target.unsqueeze(0)  # Add batch dimension if missing
        if target.ndimension() != 2:
            raise ValueError(f"Unexpected target shape: {target.shape}")

        points = points.transpose(1, 2)

        vote_pool = torch.zeros(target.size(0), target.size(1), num_part).to(device)

        for _ in range(num_votes):
            with torch.no_grad():
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

        seg_pred = vote_pool / num_votes
        cur_pred_val = seg_pred.cpu().numpy()
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((points.size(0), points.size(2))).astype(np.int32)
        target = target.cpu().numpy()

        for i in range(points.size(0)):
            cat = seg_label_to_cat[target[i, 0]]
            logits = cur_pred_val_logits[i, :, :]
            cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], axis=1) + seg_classes[cat][0]

        segmentation_results.append((points.cpu().numpy(), cur_pred_val, target))

    return segmentation_results









def render_visualization_tab(dataset):
    class_choice = st.selectbox("Select Class", list(dataset.classes.keys()))
    num_samples = st.slider(
        "Number of Samples", min_value=1, max_value=10, value=1
    )

    samples = get_class_samples(dataset,class_choice, num_samples)

    st.write(f"Displaying {num_samples} samples for class '{class_choice}'")

    for i, (point_set, cls, seg) in enumerate(samples):
        fig = px.scatter_3d(
            x=point_set[:, 0],
            y=point_set[:, 1],
            z=point_set[:, 2],
            color=seg,
            title=f"Sample {i + 1}",
        )
        st.plotly_chart(fig)

def render_segmentation_tab():
    # Checkbox to select normal channel
    normal_channel = st.checkbox('Use Normal Channel', value=False)

    # Initialize the dataset
    dataset = load_dataset(normal_channel=normal_channel)

    # Sidebar for class selection and sample size
    class_choice = st.selectbox('Select Class to Segment', list(dataset.classes.keys()))
    num_samples = st.slider('Number of Segmentation Samples', min_value=1, max_value=10, value=1)

    # Filter dataset based on selected class
    samples = get_class_samples(dataset, class_choice, num_samples)

    # Display the samples
    st.write(f"Displaying {num_samples} samples for class '{class_choice}'")

    displayed_samples = []
    for i, (point_set, cls, seg) in enumerate(samples):
        fig = px.scatter_3d(
            x=point_set[:, 0], 
            y=point_set[:, 1], 
            z=point_set[:, 2],
            color=seg,
            title=f'Sample {i + 1}'
        )
        st.plotly_chart(fig)
        displayed_samples.append((point_set, cls, seg))

    # Get segmentation results
    segmentation_results = get_segmentation_results(displayed_samples, normal_channel=normal_channel)

    # Display segmentation results
    for i, (points, seg_pred, target) in enumerate(segmentation_results):
        # Ensure points are correctly shaped for plotting
        points = points[0]  # Remove batch dimension for visualization
        fig = px.scatter_3d(
            x=points[:, 0], 
            y=points[:, 1], 
            z=points[:, 2],
            color=seg_pred,
            title=f'Segmentation Result {i + 1}'
        )
        st.plotly_chart(fig)



def main():
    tab_names = [
        "Visualization",
        "Segmentation",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        dataset = load_dataset()
        render_visualization_tab(dataset)
    with tabs[1]:
        render_segmentation_tab()

if __name__ == "__main__":
    main()
