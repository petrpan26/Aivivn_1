import numpy as np
from sklearn.model_selection import KFold
from scripts.util import f1


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
            self._models[ind].compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
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
            weights = model.get_weights()

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]


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
                model.set_weights(weights)
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])


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


class StackedGeneralizerWithHier:
    def __init__(self, models, hier_models, meta_model):
        self._models = models
        self._hier_models = hier_models

        self._meta_model = meta_model
        return

    def train_models(self, X, y, X_val, y_val, X_hier, X_hier_val, model_path, epochs, batch_size,
                     patience):
        for ind in range(len(self._models)):
            checkpoint = ModelCheckpoint(
                filepath='{}/models.hdf5'.format(model_path),
                monitor='val_f1', verbose=1,
                mode='max',
                save_best_only=True
            )
            early = EarlyStopping(monitor='val_f1', mode='max', patience=patience)
            callbacks_list = [checkpoint, early]
            self._models[ind].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
            self._models[ind].fit(
                X, y,
                validation_data=(X_val, y_val),
                callbacks=callbacks_list,
                epochs=epochs,
                batch_size=batch_size
            )
            self._models[ind].load_weights(filepath='{}/models.hdf5'.format(model_path))

        for ind in range(len(self._hier_models)):
            checkpoint = ModelCheckpoint(
                filepath='{}/models.hdf5'.format(model_path),
                monitor='val_f1', verbose=1,
                mode='max',
                save_best_only=True
            )
            early = EarlyStopping(monitor='val_f1', mode='max', patience=patience)
            callbacks_list = [checkpoint, early]
            self._hier_models[ind].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
            self._hier_models[ind].fit(
                X_hier, y,
                validation_data=(X_hier_val, y_val),
                callbacks=callbacks_list,
                epochs=epochs,
                batch_size=batch_size
            )
            self._hier_models[ind].load_weights(filepath='{}/models.hdf5'.format(model_path))

    def train_meta_model(self, X, y, X_val, y_val, X_hier, X_hier_val, model_path, epochs,
                         batch_size, patience):

        # Obtain level-1 input from each model:
        meta_input = np.zeros((len(X), len(self._models) + len(self._hier_models)))

        for ind in range(len(self._models)):
            pred = np.zeros(len(X))
            kf = KFold(n_splits=5, shuffle=False)
            model = self._models[ind]
            weights = model.get_weights()

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

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
                    validation_data=(X_val, y_val),
                    callbacks=callbacks_list,
                    epochs=epochs,
                    batch_size=batch_size
                )

                model.load_weights(filepath='{}/models.hdf5'.format(model_path))
                pred[test_index] = model.predict(X_test).reshape(-1)

                # Reset model:
                model.set_weights(weights)
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])


            meta_input[:, ind] = pred

        for ind in range(len(self._hier_models)):
            pred = np.zeros(len(X))
            kf = KFold(n_splits=5, shuffle=False)
            model = self._hier_models[ind]
            weights = model.get_weights()

            for train_index, test_index in kf.split(X):
                X_train, X_test = X_hier[train_index], X_hier[test_index]
                y_train, y_test = y[train_index], y[test_index]

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
                    validation_data=(X_hier_val, y_val),
                    callbacks=callbacks_list,
                    epochs=epochs,
                    batch_size=batch_size
                )

                model.load_weights(filepath='{}/models.hdf5'.format(model_path))
                pred[test_index] = model.predict(X_test).reshape(-1)

                # Reset model:
                model.set_weights(weights)
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])

            meta_input[:, len(self._models) + ind] = pred

        self._meta_model.fit(meta_input, y)

    def predict(self, X, X_hier):
        meta_input = self.compute_meta_data(X, X_hier)
        return (self._meta_model.predict(meta_input) > 0.5).astype(np.int8)

    def compute_meta_data(self, X, X_hier):
        prediction = np.zeros((len(X), len(self._models) + len(self._hier_models)))
        for ind in range(len(self._models)):
            pred = self._models[ind].predict(X).reshape(len(X), 1).reshape(-1)
            prediction[:, ind] = pred

        for ind in range(len(self._hier_models)):
            pred = self._hier_models[ind].predict(X_hier).reshape(len(X_hier), 1).reshape(-1)
            prediction[:, len(self._models) + ind] = pred

        return prediction

    def load_weights(self, paths, paths_hier):
        for ind in range(len(self._models)):
            self._models[ind].load_weights(paths[ind])

        for ind in range(len(self._hier_models)):
            self._hier_models[ind].load_weights(paths_hier[ind])


