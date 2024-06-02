import streamlit as st
import numpy as np
import plotly.express as px

from data_utils.ShapeNetDataLoader import PartNormalDataset


@st.cache_data()
def load_dataset(split='test', npoints=2500, normal_channel=False):
    return PartNormalDataset(
        root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', 
        npoints=npoints, 
        split=split, 
        normal_channel=normal_channel
    )


@st.cache_data()
def get_class_samples(_dataset,class_choice, num_samples):
    indices = [i for i, (cls, _) in enumerate(_dataset.datapath) if cls == class_choice]
    selected_indices = np.random.choice(indices, num_samples, replace=False)
    samples = [_dataset[i] for i in selected_indices]
    return samples


def render_visualization_tab(dataset):
    class_choice = st.sidebar.selectbox('Select Class', list(dataset.classes.keys()))
    num_samples = st.sidebar.slider('Number of Samples', min_value=1, max_value=10, value=5)

    samples = get_class_samples(dataset,class_choice, num_samples)


    st.write(f"Displaying {num_samples} samples for class '{class_choice}'")


    for i, (point_set, cls, seg) in enumerate(samples):
        fig = px.scatter_3d(
            x=point_set[:, 0], 
            y=point_set[:, 1], 
            z=point_set[:, 2],
            color=seg,
            title=f'Sample {i + 1}'
        )
        st.plotly_chart(fig)

def main():
    dataset = load_dataset()

    tab_names = [
        "Visualization",
        "T-Net",
        "Global Features",
        "Local Features",
        "Classification",
        "Segmentation",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_visualization_tab(dataset)
    with tabs[1]:
        st.write("T-Net")
    with tabs[2]:
        st.write("Global Features")
    with tabs[3]:
        st.write("Local Features")
    with tabs[4]:
        # visualize_pointnet_results(dataset, task='classification')
        st.write("Classification")

if __name__ == '__main__':
    main()