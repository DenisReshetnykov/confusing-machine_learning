import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import scipy as sc

# For preprocessing the data
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.model_selection import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score


def load_csv_data(datafile):
    loaded_data = pd.read_csv(datafile)
    return loaded_data
loaded_data=load_csv_data('EEG_data.csv')
# loaded_data=load_csv_data('EEG_data_unclear.csv')


def statistic_for_features(data):
    for value in data.columns.values[2:13]:
        print(value+' is zero: '+str(len(data.where(data[value] == 0).dropna())))
        print(value+' max: '+str(data[value].max())+
              ' min: '+str(data[value].min())+
              ' average: '+ str(np.mean(data[value]))+
              ' median: '+str(np.median(data[value])))


def get_data_by_student_and_videoframe(data, student_id, videoframe_id):
    trunkated_data = data.where(data['SubjectID'] == float(student_id)).where(
        data['VideoID'] == float(videoframe_id)).dropna()
    return trunkated_data[0:112]


def repack_data_to_same_size_dataframes_per_block(data):
    repacked_data = pd.DataFrame()
    for student in range(9):
        for videoframe in range(10):
            df = get_data_by_student_and_videoframe(data, student, videoframe)
            repacked_data = repacked_data.append(df)
    return repacked_data.reset_index(drop=True)
repacked_data = repack_data_to_same_size_dataframes_per_block(loaded_data)

def restore_missed_data(data):
    for n in range(112):
        data.iloc[3360+336+n, 3] = np.mean([data.iloc[1120*i+336+n, 3] for i in [0, 1, 2, 4, 5, 6, 7, 8]])
        data.iloc[3360 + 336 + n, 2] = np.mean([data.iloc[1120 * i + 336 + n, 2] for i in [0, 1, 2, 4, 5, 6, 7, 8]])
    return data
restored_data = restore_missed_data(repacked_data)

def anti_aliasing_data(data, window):
    for feature in range(4,13):
        for student in range(9):
            for video in range(10):
                for n in range(112):
                    position = student*1120+video*112+n
                    data.iloc[position, feature] = \
                            np.median(data.iloc[position-min(window, n):position+min(window, 112-n), feature])
        print('AA for '+str(data.columns.values[feature]+' ended'))
    return data
aa_data = anti_aliasing_data(restored_data, 5)
# print(aa_data)

def standartized_data(data):
    scaler = StandardScaler()
    scaler.fit(data.iloc[:, 2:13])
    data.iloc[:, 2:13] = scaler.transform(data.iloc[:, 2:13])
    return data
# std_data = standartized_data(restored_data)


def plot_features_boxplots(data):
    axes = data.iloc[:, 2:13].plot(kind='box',
                            subplots=True,
                            layout=(2, 6),
                            sharex=False,
                            sharey=False,
                            whis=[0, 95],
                            sym='.',
                            widths=[0.75])
    plt.subplots_adjust(wspace = 1)
    plt.show()
# plot_features_boxplots(std_data)


def plot_features_corelation(data):
    feature_columns = data.columns.values[2:13]
    colors = ['red', 'blue']
    axis = pd.plotting.scatter_matrix(data.iloc[:, 2:13],
                                      c=data.iloc[:, 13].apply(lambda x: colors[int(x)]),
                                      alpha=0.1,
                                      figsize=(10, 10),
                                      range_padding=0.01,
                                      diagonal='hist')
    plt.subplots_adjust()
    axis[0,5].set_title('Scatterplot Matrix')
    feature_iterator = iter(feature_columns)
    for ax in axis[:,0]:
        ax.set_yticks([])
        ax.set_ylabel(next(feature_iterator), fontsize=6)
    feature_iterator = iter(feature_columns)
    for ax in axis[-1,:]:
        ax.set_xticks([])
        ax.set_xlabel(next(feature_iterator), fontsize=6)
    # plt.show()
    plt.savefig('scatterplot-matrix-std.png', format='png')
    plt.close()
# plot_features_corelation(std_data)

def correlation_matrix(data):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 20)
    cax = ax1.imshow(data.iloc[:, 2:13].corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Feature Correlation')
    labels=data.columns.values[2: 13]
    ax1.set_yticks(ticks=range(0, 11))
    ax1.set_xticks(ticks=range(0, 11))
    ax1.set_xticklabels(labels, rotation='vertical', fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax)
    plt.savefig('correlation-matrix-AA5.png', format='png')
# correlation_matrix(aa_data)

def plot_features(data):
    feature_columns = data.columns.values[2:13]
    plt.title('Features')
    for feature in range(len(feature_columns)):
        plt.subplot(len(feature_columns), 1, feature+1)
        for student in range(9):
            plt.plot(data[feature_columns[feature]][1120*student: 1120*student+560],
                     color = 'red', linestyle = '', marker = '.', ms = 0.2)
            plt.plot(data[feature_columns[feature]][1120*student+560: 1120*(student+1)],
                     color = 'blue', linestyle = '', marker = '.', ms = 0.2)
        plt.ylabel(feature_columns[feature], rotation=0, labelpad=30,)
        plt.yticks([],[])
        plt.xticks(range(112*5, 112*5*2*9+1, 112*5*2), ['S' + str(n) for n in range(1, 10)])
    plt.savefig('Features-AA5.png', format='png')
    plt.close()
# plot_features(std_data)

def create_test_frame(data):
    x = data.iloc[:, 2:13]
    return x
# print(create_test_frame(repack_data_to_same_size_dataframes_per_block(load_csv_data('EEG_data.csv'))))


def create_user_defined_label_vector(data):
    y = data['user-definedlabeln'].astype(int)
    return y
# print(create_user_defined_label_vector(repack_data_to_same_size_dataframes_per_block(load_csv_data('EEG_data.csv'))))


def create_user_predefined_label_vector(data):
    y = data['predefinedlabel'].astype(int)
    return y
# print(create_user_predefined_label_vector(repack_data_to_same_size_dataframes_per_block(load_csv_data('EEG_data.csv'))))


def create_mean_feature_data_frame(data):
    x_mean = np.zeros((90, 11))
    for student in range(9):
        for videoframe in range(10):
            dataframe = get_data_by_student_and_videoframe(data, student, videoframe)
            x_mean[student * 10 + videoframe] = np.mean(dataframe, axis=0)[2:13]
    return x_mean


def gaussian_clasifier(x_model, y_model, x_test):
    model = GaussianNB()
    model.fit(x_model, y_model)
    predicted = model.predict(x_test)
    return predicted


def calculate_accuracy(predicted_data, test_data):
    accuracy = 0.0
    if len(predicted_data) == len(test_data):
        for index in range(len(predicted_data)):
            if predicted_data[index] == test_data[index]:
                accuracy += 1
        accuracy /= len(predicted_data)
    return accuracy


def calculate_cross_validated_student_independent_clasifier_acuracy(data, chunk_size):
    clasifier_accuracy = np.zeros(9)
    x = create_test_frame(data)
    y = create_user_predefined_label_vector(data)
    for student in range(9):
        x_model = np.append(x[0:student*10*chunk_size], x[(student+1)*10*chunk_size:], axis=0)
        y_model = np.append(y[0:student*10*chunk_size], y[(student+1)*10*chunk_size:], axis=0)
        x_test = x[student*10*chunk_size:(student+1)*10*chunk_size]
        y_test = np.array(y[student * 10 * chunk_size:(student + 1) * 10*chunk_size], dtype=int)
        predicted = gaussian_clasifier(x_model, y_model, x_test)
        clasifier_accuracy[student] = round(calculate_accuracy(predicted, y_test), 3)
    return clasifier_accuracy


def calculate_cross_validated_student_specific_clasifier_acuracy(data, chunk_size):
    clasifier_accuracy = np.zeros(9)
    x = create_test_frame(data)
    y = create_user_predefined_label_vector(data)
    for student in range(9):
        student_accuracy = np.zeros(10)
        for video in range(10):
            x_model = np.append(x[0:student * chunk_size * 10 + chunk_size * video],
                                x[student * chunk_size * 10 + chunk_size * (video + 1):], axis=0)
            y_model = np.append(y[0:student * chunk_size * 10 + chunk_size * video],
                                y[student * chunk_size * 10 + chunk_size * (video + 1):], axis=0)
            x_test = x[student * chunk_size * 10 + chunk_size * video:
                       student * chunk_size * 10 + chunk_size * (video + 1)]
            y_test = np.array(y[student * chunk_size * 10 + chunk_size * video:
                       student * chunk_size * 10 + chunk_size * (video + 1)], dtype=int)
            predicted = gaussian_clasifier(x_model, y_model, x_test)
            student_accuracy[video] = round(calculate_accuracy(predicted, y_test), 3)
        clasifier_accuracy[student] = round(np.mean(student_accuracy), 3)
    return clasifier_accuracy


def plot_accuracy_histogram(data):
    sica = calculate_cross_validated_student_independent_clasifier_acuracy(data, 112)
    ssca = calculate_cross_validated_student_specific_clasifier_acuracy(data, 112)
    print('sica: '+str(sica))
    print('ssca: '+str(ssca))
    studentlist = ['S' + str(n) for n in range(1, 10)]
    plt.bar(np.arange(10), np.append(round(np.mean(sica), 3), sica), 0.4, color='green', label='Student independent')
    plt.bar(np.arange(10)+0.4, np.append(round(np.mean(ssca), 3), ssca), 0.4, color='blue', label='Student specific')
    plt.xticks(np.arange(10), ['Average']+studentlist)
    plt.yticks(np.arange(0, 1.01, 0.1), [str(n)+'%' for n in range(0, 101, 10)])
    plt.grid(axis='y')
    plt.legend()
    plt.savefig('pre-defined accuracy-AA5.png', format='png')
    plt.close()
plot_accuracy_histogram(aa_data)