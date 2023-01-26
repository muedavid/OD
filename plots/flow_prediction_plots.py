import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def get_flow(flow, numpy=False):
    if numpy:
        x_flow = np.abs(flow[:, :, :, 0])
        y_flow = np.abs(flow[:, :, :, 1])
    else:
        x_flow = np.abs((flow[:, :, :, 0]).numpy())
        y_flow = np.abs((flow[:, :, :, 1]).numpy())
    return x_flow, y_flow


def plot_flow_field(images=None, prior=None, flow_ground_truth=None, flow_prediction=None, batch_size=0, num_exp=None):
    flow_max = 4.0
    
    if num_exp:
        num_exp = min(num_exp, batch_size)
    else:
        num_exp = batch_size
    
    rows = 1 + (prior is not None) + 2 * (flow_ground_truth is not None) + 2 * (flow_prediction is not None)
    cols = num_exp
    plt.figure(figsize=(8 * cols, 8 * rows))
    for i in range(num_exp):
        plt.subplot(rows, cols, i + 1)
        plt.title("Images")
        plt.imshow(tf.keras.preprocessing.image.array_to_img(images[i, :, :, :]))
        plt.axis('off')
        if prior is not None:
            plt.subplot(rows, num_exp, num_exp + i + 1)
            plt.title("Prior Edge Maps")
            plt.imshow(tf.keras.preprocessing.image.array_to_img(prior[i, :, :, :]))
            plt.axis('off')
        if flow_ground_truth is not None:
            x_flow_gt, y_flow_gt = get_flow(flow_ground_truth)
            
            plt.subplot(rows, num_exp, (1 + (prior is not None)) * num_exp + i + 1)
            plt.title("Ground Truth Flow x Direction")
            plt.imshow(x_flow_gt[i, :, :], cmap='gray', vmin=0, vmax=flow_max)
            plt.axis('off')
            plt.subplot(rows, num_exp, (1 + (prior is not None)) * num_exp + num_exp + i + 1)
            plt.title("Ground Truth Flow y Direction")
            plt.imshow(y_flow_gt[i, :, :], cmap='gray', vmin=0, vmax=flow_max)
            plt.axis('off')
        if flow_prediction is not None:
            x_flow_pred, y_flow_pred = get_flow(flow_prediction, numpy=True)
            
            plt.subplot(rows, num_exp,
                        (1 + (prior is not None) + 2 * (flow_ground_truth is not None)) * num_exp + i + 1)
            plt.title("Flow x Direction")
            plt.imshow(x_flow_pred[i, :, :], cmap='gray', vmin=0, vmax=flow_max)
            plt.axis('off')
            plt.subplot(rows, num_exp,
                        (1 + (prior is not None) + 2 * (flow_ground_truth is not None)) * num_exp + num_exp + i + 1)
            plt.title("Flow y Direction")
            plt.imshow(y_flow_pred[i, :, :], cmap='gray', vmin=0, vmax=flow_max)
            plt.axis('off')
