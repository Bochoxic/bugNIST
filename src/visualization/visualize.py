from skimage import io
import numpy as np
import plotly.graph_objects as go   

def visualize_3D(image_tif:np.ndarray):
    """Visualize in 3D the volume. It needs to open the visualization in a web app.

    Keyword arguments:
    image_tiff -- the volumetric image opened with skimage.io.imread
    """
    X, Y, Z = np.mgrid[-8:8:complex(0,image_tif.shape[0]), -8:8:complex(0,image_tif.shape[1]), -8:8:complex(0,image_tif.shape[2])]
    fig = go.Figure(data=go.Volume(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=image_tif.flatten(),
                    isomin=1,
                    isomax=255,
                    opacity=0.05, # needs to be small to see through all surfaces
                    surface_count=17, # needs to be a large number for good volume rendering
    ))
    fig.show()


def main():

    # Read the image
    in_dir = "data/processed/bugnist_128_split/test/AC/"
    im_name = "bcrick 11_000.tif"
    image_tif = io.imread(in_dir + im_name)
    visualize_3D(image_tif)

if __name__ == "__main__":
    main()