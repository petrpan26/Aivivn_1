import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

from scripts.util import read_file, tokenize, make_embedding, text_to_sequences
from sklearn.model_selection import train_test_split


from keras.callbacks import EarlyStopping, ModelCheckpoint


class StackedGeneralizer:

    def __init__(self, models, meta_model):
        self._models = models
        self._meta_model = meta_model
        return


    def train_models(self, X, y, X_val, y_val, model_path, epochs, batch_size, patience):
        for ind in range(len(self._models)):
            checkpoint = ModelCheckpoint(
                filepath='{}/models.hdf5'.format(model_path),
                monitor='val_f1', verbose=1,
                mode='max',
                save_best_only=True
            )
            early = EarlyStopping(monitor='val_f1', mode='max', patience=patience)
            callbacks_list = [checkpoint, early]
            self._models[ind].fit(
                X, y,
                validation_data= (X_val, y_val),
                callbacks=callbacks_list,
                epochs=epochs,
                batch_size=batch_size
            )
            self._models[ind].load_weights(filepath='{}/models.hdf5'.format(model_path))



    def train_meta_model(self, X, y, X_val, y_val, model_path, epochs, batch_size, patience):

        # Obtain level-1 input from each model:
        meta_input = np.zeros((len(X), len(self._models)))

        for ind in range(len(self._models)):
            pred = np.zeros(len(X))
            kf = KFold(n_splits = 5, shuffle = False)
            model = self._models[ind]

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model.save_weights(filepath='{}/dumped.hdf5'.format(model_path))
                checkpoint = ModelCheckpoint(
                    filepath='{}/models.hdf5'.format(model_path),
                    monitor='val_f1', verbose=1,
                    mode='max',
                    save_best_only=True
                )
                early = EarlyStopping(monitor='val_f1', mode='max', patience=patience)
                callbacks_list = [checkpoint, early]
                model.fit(
                    X_train, y_train,
                    validation_data= (X_val, y_val),
                    callbacks=callbacks_list,
                    epochs=epochs,
                    batch_size=batch_size
                )

                model.load_weights(filepath='{}/models.hdf5'.format(model_path))
                pred[test_index] = model.predict(X_test).reshape(-1)

                # Reset model:
                model.load_weights(filepath='{}/dumped.hdf5'.format(model_path))

            meta_input[:, ind] = pred


        self._meta_model.fit(meta_input, y)


    def predict(self, X):
        meta_input = self.compute_meta_data(X)
        return (self._meta_model.predict(meta_input) > 0.5).astype(np.int8)


    def compute_meta_data(self, X):
        prediction = np.zeros((len(X), len(self._models)))
        for ind in range(len(self._models)):
            pred = self._models[ind].predict(X).reshape(len(X), 1).reshape(-1)
            prediction[:, ind] = pred

        return prediction

    def load_weights(self, paths):
        for ind in range(len(self._models)):
            self._models[ind].load_weights(paths[ind])
