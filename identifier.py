# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 16:04:08 2021

@author: mszx2
"""
from sklearn.manifold import TSNE
from itertools import islice, chain
from sklearn.base import ClassifierMixin, BaseEstimator
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from nltk.util import everygrams
from matplotlib.patches import Patch
from sklearn.model_selection import (KFold, ShuffleSplit,train_test_split,
                                     cross_val_score)


class TorHostIdentification(BaseEstimator, ClassifierMixin):
    def __init__(self, model, n_neighbors = 2):
        self.model = model
        self.n_neighbors = n_neighbors
        self.knn = None
    
    def show_cluster(self, df = None, color_class = [], filename = 'sns.png'):
        _df = [self.__get_vector(word) for word in df]
        tsne_em = TSNE(n_components=2,
                       perplexity=30.0,
                       n_iter=300,
                       verbose=1).fit_transform(_df)
        df_subset = {}
        df_subset['tsne-2d-one'] = tsne_em[:, 0]
        df_subset['tsne-2d-two'] = tsne_em[:, 1]
        df_subset['Маркеры классов'] = color_class
        
        plt.figure(figsize=(8, 6))
        marks = {0: "s", 1: "X"}
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="Маркеры классов",
            data=df_subset,
            legend="full",
            alpha=0.7,
            style="Маркеры классов",
            markers=marks
        )
        plt.show()
        self.show_knn(tsne_em, color_class, 'knn_' + filename)

    def show_knn(self, X, y, filename='knn.png'):
        h = 10
        cmap_light = ListedColormap(['orange', 'cyan'])
        cmap_bold = ['darkorange', 'c']
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                       weights='distance')
        knn.fit(X, y)
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        sns.scatterplot(x=X[:, 0], y=X[:, 1],
                        style=['Настоящие хосты', 'Хосты анонимной сети'] ,
                    palette=cmap_bold, alpha=1.0, edgecolor="black")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Классификация по k ближайших соседей (k = %i)"
              % (self.n_neighbors))
        plt.xlabel('корреляционная координата (Ox)')
        plt.ylabel('корреляционная координата (Oy)')
        plt.show()
        #plt.savefig(filename)

    def __infer_vector(self, word):
        if word in self.model.index_to_key:
            return self.model[word]
        else:
            return np.zeros(300)

    def __get_vector(self, host, is_bin = False):
        if is_bin:
            return self.model.wv.get_vector(host)
        test = self.__infer_vector(host)
        if np.count_nonzero(test) > 0:
            return test
        return np.sum([self.__infer_vector("".join(word)) for word in 
                    everygrams(host, min_len=2, max_len=6)], axis=0)

    def fit(self, X, y):
       self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                       weights='distance')
       _X = [self.__get_vector(word) for word in X]
       self.knn.fit(_X, y)
       return self
   
    def predict(self, X):
        _X = [self.__get_vector(word) for word in X]
        return self.knn.predict(_X)
    
    def predict_proba(self, X):
        pred = self.predict(X)
        return pred / np.sum(pred, axis=1)[:, np.newaxis]


def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    cmap_data = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['класс', 'группа']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Выборка индекса', ylabel="Итерацияя кроссвалидации",
           ylim=[n_splits+2.2, -.2], xlim=[0, len(X)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax

def cross_validation_data_show(X, y, cvs = [KFold],
                          n_splits=5):
    cmap_cv = plt.cm.coolwarm
    uneven_groups = np.sort(np.random.randint(0, 10, len(X)))
    cvs_objs = []
    for cv in cvs:
        this_cv = cv(n_splits=n_splits)
        cvs_objs.append(this_cv)

        fig, ax = plt.subplots(figsize=(6, 3))
        plot_cv_indices(this_cv, X, y, uneven_groups, ax, n_splits)

        ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
              ['Тестируемое множество', 'Обучающее множество'], loc=(1.02, .8))
        plt.tight_layout()
        fig.subplots_adjust(right=.7)
    plt.show()
    return cvs_objs

def cv_scores(clf, X, y, cvs):
    scores = []
    for cv in cvs:
        score = cross_val_score(clf, X, y, cv=cv)
        scores.append(score)
        print("Процент угаданных меток классов на итерациях кроссвалидации:")
        print(list(map(lambda s: '{}%'.format(s * 100), score)))
        print("Средний процент: {}%".format(np.mean(score)))
    return scores

def test(score):
    print("Процент угаданных меток классов на итерациях кроссвалидации:")
    print(list(map(lambda s: '{}%'.format(s * 100), score)))
    print("Средний процент: {}%".format(100 * np.mean(score)))

def test_for(model, mixed_host_generator, sizes):
    identifier = TorHostIdentification(model,
                                       n_neighbors=3)
    for size in sizes:
        print('-' * 85)
        X, y = zip(*islice(mixed_host_generator, size))
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)
        
        #param_grid = {‘n_neighbors’: np.arange(1, 25)}
        #use gridsearch to test all values for n_neighbors
        #knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
        #knn_gscv.best_params_
        identifier.fit(X_train, y_train)
        predict = identifier.predict(X_test)
        cvs = cross_validation_data_show(X_train, y_train)
        cv_scores(identifier, X_test, y_test, cvs)
        th = int(len(X_test) * 15 / 100)
        result = zip(X_test[:th], predict[:th])
        print('Тестируемое множество (15%) в формате -> (хост, предсказание)')
        print(list(result))
        identifier.show_cluster(X_test, predict)
    return identifier, X_test, y_test, X_train, y_train, predict
