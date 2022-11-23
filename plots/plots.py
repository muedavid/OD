import matplotlib.pyplot as plt


def plot_training_history(history=None, list_of_loss_names=None, list_of_metric_names=None, history_fine=None,
                          epochs=None, save=False, path=None,
                          max_f1=None):
    dim = (int(len(list_of_loss_names + list_of_metric_names) / 2), 2)
    if history_fine is None:
        x_axis_labels = range(1, len(history[list_of_loss_names[0]]) + 1)
    else:
        x_axis_labels = range(1, len(history[list_of_loss_names[0]] + history_fine[list_of_loss_names[0]]) + 1)
    
    fig = plt.figure(figsize=(dim[0] * 12, dim[1] * 6))
    
    if max_f1 is not None:
        fig.suptitle("Maximum F1 Score = {:.3f} at threshold = {:.3f}".format(max_f1[0], max_f1[1]))
    
    # plot loss history
    for i in range(len(list_of_loss_names)):
        plt.subplot(dim[0], dim[1], i + 1)
        
        if history_fine is None:
            plt.plot(x_axis_labels, history[list_of_loss_names[i]], label='Training ' + list_of_loss_names[i])
            plt.plot(x_axis_labels, history["val_" + list_of_loss_names[i]],
                     label='Validation ' + list_of_loss_names[i])
            plt.xticks(x_axis_labels)
        else:
            plt.plot(x_axis_labels, history[list_of_loss_names[i]] + history_fine[list_of_loss_names[i]],
                     label='Training ' + list_of_loss_names[i])
            plt.plot(x_axis_labels, history["val_" + list_of_loss_names[i]] +
                     history_fine["val_" + list_of_loss_names[i]], label='Validation ' + list_of_loss_names[i])
            plt.xticks(x_axis_labels)
            
            plt.plot([epochs, epochs], plt.ylim(), label='Start Fine Tuning')
        
        plt.legend(loc='upper right')
        plt.ylabel(list_of_loss_names[i])
        plt.xlabel('epoch')
    
    # plot metrics history
    for i in range(len(list_of_metric_names)):
        plt.subplot(dim[0], dim[1], i + 1 + len(list_of_loss_names))
        
        if "accuracy" in list_of_metric_names[i]:
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])
        if history_fine is None:
            plt.plot(x_axis_labels, history[list_of_metric_names[i]], label='Training ' + list_of_metric_names[i])
            plt.plot(x_axis_labels, history["val_" + list_of_metric_names[i]],
                     label='Validation ' + list_of_metric_names[i])
            plt.xticks(x_axis_labels)
        else:
            plt.plot(x_axis_labels, history[list_of_metric_names[i]] + history_fine[list_of_metric_names[i]],
                     label='Training ' + list_of_metric_names[i])
            plt.plot(x_axis_labels,
                     history["val_" + list_of_metric_names[i]] + history_fine["val_" + list_of_metric_names[i]],
                     label='Validation ' + list_of_metric_names[i])
            
            plt.xticks(x_axis_labels)
            plt.plot([epochs, epochs], plt.ylim(), label='Start Fine Tuning')
        
        plt.legend(loc='lower right')
        plt.ylabel(list_of_metric_names[i])
        plt.xlabel('epoch')
        
        if save:
            plt.savefig(path, bbox_inches='tight')
    
    plt.draw()
    plt.show()
