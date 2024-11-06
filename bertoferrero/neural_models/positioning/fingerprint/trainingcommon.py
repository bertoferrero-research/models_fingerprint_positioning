# Copyright 2024 Alberto Ferrero López
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import math
import os
import sys
import shutil
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Union

# region Carga de datos


def load_data(data_file: str, scaler_file: str = None, train_scaler_file: bool = False, include_pos_z: bool = True, scale_y: bool = False, remove_not_full_rows: bool = False, not_valid_sensor_value: int = None, return_valid_sensors_map: bool = False):
    # Cargamos los ficheros
    data = pd.read_csv(data_file)

    # Preparamos los datos
    X, y = prepare_data(data, include_pos_z, scale_y, remove_not_full_rows)

    # Preparamos el mapa de sensores no válidos
    Xmap = None
    if not_valid_sensor_value is not None and return_valid_sensors_map:
        Xmap = X.ne(not_valid_sensor_value)
        Xmap = Xmap.astype(np.int32)
        # Tambien se podría haber hecho directamente Xmap = X != 100

    # Escalamos
    if scaler_file is not None:
        if train_scaler_file:
            X = scale_RSSI_training(scaler_file, X)
        else:
            X = scale_RSSI_track(scaler_file, X)
        X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Devolvemos
    if Xmap is not None:
        return X, y, Xmap
    return X, y


def load_data_inverse(data_file: str, scaler_file: str, train_scaler_file: bool = False, include_pos_z: bool = True, scale_y: bool = False, remove_not_full_rows: bool = False, separate_mac_and_pos: bool = False):
    '''
    Devuelve los datos de entrenamiento y test preparados para entrenar la predicción de posicion a rssi.
    La salida para X contendrá tres columnas, pos_x, pos_y y mac
    La salida para y contendrá el valor rssi correspondiente
    '''
    # Recogemos los valores originales
    X, y = load_real_track_data(
        data_file, None, include_pos_z, scale_y, remove_not_full_rows)

    # Acumulamos los datos en el formato nuevo
    sensors = X.columns.to_list()
    sensors.sort()
    data = []
    for index, row in y.iterrows():
        for i in range(len(sensors)):
            sensor = sensors[i]
            data.append({
                'pos_x': row['pos_x'],
                'pos_y': row['pos_y'],
                'sensor_mac': i,
                'rssi': X[sensor][index]
            })
    data = pd.DataFrame(data)

    # Convertimos los dtype
    data['pos_x'] = pd.to_numeric(data['pos_x'], downcast='float')
    data['pos_y'] = pd.to_numeric(data['pos_y'], downcast='float')
    data['sensor_mac'] = pd.to_numeric(data['sensor_mac'], downcast='integer')
    data['rssi'] = pd.to_numeric(data['rssi'], downcast='float')

    # Dividimos X e y
    X = data[['pos_x', 'pos_y', 'sensor_mac']]
    y = data[['rssi']]

    # Escalamos el rssi
    if train_scaler_file:
        y = scale_RSSI_training(scaler_file, y)
    else:
        y = scale_RSSI_track(scaler_file, y)

    if separate_mac_and_pos:
        X = [X[['pos_x', 'pos_y']], X[['sensor_mac']]]

    # Devolvemos
    return X, y, sensors


def prepare_data(data, include_pos_z: bool = True, scale_y: bool = False, remove_not_full_rows: bool = False):
    # Eliminamos las filas que no tienen todos los datos

    if remove_not_full_rows:
        # Reemplazamos los -200 en las columnas de rssi (a partir de la cuarta) por NaN
        data.iloc[:, 4:] = data.iloc[:, 4:].replace(-200, np.nan)

    # Extraemos cada parte
    y = data.iloc[:, 1:(4 if include_pos_z else 3)]
    X = data.iloc[:, 4:]
    # Escalamos y
    if scale_y:
        y = scale_dataframe(y)

    # Ordenamos alfabéticamente las columnas de X, asegurandonos de que todos los datasets van en el mismo orden
    X = X.reindex(sorted(X.columns), axis=1)

    # Inputamos los valores de -200 por el modelo
    # YA no es necesario, hay un csv con los datos inputados
    # if remove_not_full_rows:
    #    X = imputing_predict_na_data(X)

    # Convertimos a float32 e in32 para reducir complejidad
    # y = y.astype(np.float32)
    # X = X.astype(np.int32)

    # Devolvemos
    return X, y


def imputing_predict_na_data(data: pd.DataFrame):
    # Creamos una copia de data para el trabajo
    data_tmp = data.copy()
    # Reemplazamos los -200 por NaN
    data = data.replace(-200, np.nan)
    # Borramos las filas que tengan 3 o mas NaN
    # data_tmp = data_tmp.dropna(thresh=3)
    # Nos aseguramos que en el tmp sea -200
    data_tmp = data_tmp.replace(np.nan, -200)

    # Cargamos el scaler
    scaler = MinMaxScaler()
    scaler.fit([[-100], [0]])
    # Escalamos
    for sensor in data_tmp.columns:
        data_tmp[sensor] = scaler.transform(
            data_tmp[sensor].values.reshape(-1, 1)).flatten()

    # Cargamos el modelo
    model = tf.keras.models.load_model(
        'dataset_imputing_values/files/model.h5')
    # Predecimos
    data_tmp_output = model.predict(data_tmp.values.reshape(
        data_tmp.shape[0], data_tmp.shape[1], 1))
    # Desescalamos
    data_tmp_output = scaler.inverse_transform(data_tmp_output.reshape(
        data_tmp_output.shape[0], data_tmp_output.shape[1]))
    data_tmp_output = pd.DataFrame(
        data_tmp_output, columns=data_tmp.columns).round()

    # Reemplazamos los -200 por los valores predichos
    for index, row in data.iterrows():
        for sensor in data.columns:
            if np.isnan(row[sensor]):
                data[sensor][index] = data_tmp_output[sensor][index]

    # Devolvemos
    return data


# endregion

# region escalado de datos

def scale_RSSI_training(scaler_file: str, X_data):
    scaler = StandardScaler()
    scaler.fit(X_data)
    os.makedirs(os.path.dirname(scaler_file), exist_ok=True)
    with open(scaler_file, 'wb') as scalerFile:
        pickle.dump(scaler, scalerFile)
        scalerFile.close()

    X_data_scaled = X_data.copy()
    X_data_scaled[X_data.columns] = scaler.transform(X_data)
    X_data = X_data_scaled

    return X_data


def scale_RSSI_track(scaler_file: str, X):
    os.makedirs(os.path.dirname(scaler_file), exist_ok=True)
    with open(scaler_file, 'rb') as scalerFile:
        scaler = pickle.load(scalerFile)
        scalerFile.close()

    X_scaled = X.copy()
    X_scaled[X.columns] = scaler.transform(X)
    X = X_scaled

    return X


def get_scaler_pos_x():
    """
    Devuelve un scaler para la posición x
    Returns: 
        MinMaxScaler    
    """
    scaler = MinMaxScaler()
    scaler.fit([[0], [20.660138018121128]])
    return scaler


def get_scaler_pos_y():
    """
    Devuelve un scaler para la posición y
    Returns: 
        MinMaxScaler    
    """
    scaler = MinMaxScaler()
    scaler.fit([[0], [17.64103475472807]])
    return scaler


def scale_pos_x(pos_x: pd.Series):
    """
    Escala la posición x
    Args:
        pos_x (pd.Series): posición x
    Returns:
        pd.Series: posición x escalada
    """
    scaler = get_scaler_pos_x()
    return scaler.transform(pos_x.values.reshape(-1, 1)).flatten()


def scale_pos_x_single(pos_x: float):
    """
    Escala la posición x
    Args:
        pos_x (float): posición x
    Returns:
        float: posición x escalada
    """
    scaler = get_scaler_pos_x()
    return scaler.transform([[pos_x]])[0][0]


def scale_pos_y(pos_y: pd.Series):
    """
    Escala la posición y
    Args:
        pos_y (pd.Series): posición y
    Returns:
        pd.Series: posición y escalada
    """
    scaler = get_scaler_pos_y()
    return scaler.transform(pos_y.values.reshape(-1, 1)).flatten()


def scale_pos_y_single(pos_y: float):
    """
    Escala la posición y
    Args:
        pos_y (float): posición y
    Returns:
        float: posición y escalada
    """
    scaler = get_scaler_pos_y()
    return scaler.transform([[pos_y]])[0][0]


def scale_dataframe(data: pd.DataFrame):
    """
    Escala un dataframe
    Args:
        data (pd.DataFrame): dataframe a escalar
    Returns:
        pd.DataFrame: dataframe escalado
    """
    data_scaled = data.copy()
    data_scaled['pos_x'] = scale_pos_x(data['pos_x'])
    data_scaled['pos_y'] = scale_pos_y(data['pos_y'])
    return data_scaled


def descale_pos_x(pos_x: pd.Series):
    """
    Desescala la posición x
    Args:
        pos_x (pd.Series): posición x
    Returns:
        pd.Series: posición x desescalada
    """
    scaler = get_scaler_pos_x()
    return scaler.inverse_transform(pos_x.values.reshape(-1, 1)).flatten()


def descale_pos_x_np(pos_x: np.ndarray):
    """
    Descales the given position values using the scaler for pos_x.

    Parameters:
    pos_x (np.ndarray): The position values to be descaled.

    Returns:
    np.ndarray: The descaled position values.
    """
    scaler = get_scaler_pos_x()
    return scaler.inverse_transform(pos_x.reshape(-1, 1)).flatten()


def descale_pos_y(pos_y: pd.Series):
    """
    Desescala la posición y
    Args:
        pos_y (pd.Series): posición y
    Returns:
        pd.Series: posición y desescalada
    """
    scaler = get_scaler_pos_y()
    return scaler.inverse_transform(pos_y.values.reshape(-1, 1)).flatten()


def descale_pos_y_np(pos_y: np.ndarray):
    """
    Descales the given position values using the scaler for pos_y.

    Parameters:
    pos_y (np.ndarray): The position values to be descaled.

    Returns:
    np.ndarray: The descaled position values.
    """
    scaler = get_scaler_pos_y()
    return scaler.inverse_transform(pos_y.reshape(-1, 1)).flatten()


def descale_dataframe(data: pd.DataFrame):
    """
    Desescala un dataframe
    Args:
        data (pd.DataFrame): dataframe a desescalar
    Returns:
        pd.DataFrame: dataframe desescalado
    """
    data_scaled = data.copy()
    data_scaled['pos_x'] = descale_pos_x(data['pos_x'])
    data_scaled['pos_y'] = descale_pos_y(data['pos_y'])
    return data_scaled


def descale_numpy(data: np.ndarray):
    """
    Descales the input numpy array by applying descaling functions to the x and y coordinates.

    Args:
        data (np.ndarray): The input numpy array.

    Returns:
        np.ndarray: The descaled numpy array.
    """
    data[:, 0] = descale_pos_x_np(data[:, 0])
    data[:, 1] = descale_pos_y_np(data[:, 1])
    return data

# endregion

# region transformación de datos


def posXYlist_to_grid(pos_list: np.array, cell_amount_x: int, cell_amount_y: int, max_position_x: float = 20.660138018121128, max_position_y: float = 17.64103475472807, use_caregorical_output: bool = True):
    '''
    Transforma una lista de posiciones x, y al identificador de celda correspondiente
    Se enumeran las celdas como {x_i},{y_i}, siendo la primera celda la 0,0.
    Si se usa el output categórico, se devuelve un array de celdas numeradas a partir del 0, incrementandose el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    La numeración partirá del origen de coordenadas y se incrementará siempre hacia la x primero y luego hacia la y
    Args:
        pos_list (np.array): lista de posiciones x, y sin escalar
        cell_amount_x (int): Número de celdas en el eje x
        cell_amount_y (int): Número de celdas en el eje y
        max_position_x (float): Valor máximo de la posición x
        max_position_y (float): Valor máximo de la posición y
        use_caregorical_output (bool): Si se activa, devuelve un array de celdas numeradas a partir del 0, incrementandose el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    Returns:
        np.array: identificadores de celda en el formato indicado
    '''
    result = [posXY_to_grid(pos_x, pos_y, cell_amount_x, cell_amount_y, max_position_x, max_position_y,
                            use_caregorical_output=use_caregorical_output) for pos_x, pos_y in pos_list]

    return np.array(result)


def posXY_to_grid(pos_x: float, pos_y: float, cell_amount_x: int, cell_amount_y: int, max_position_x: float = 20.660138018121128, max_position_y: float = 17.64103475472807, use_caregorical_output: bool = True):
    '''
    Transforma una posición x, y al identificador de celda correspondiente
    Se enumeran las celdas como {x_i},{y_i}, siendo la primera celda la 1,1.
    Si se usa el output categórico, se devuelve un array de celdas numeradas a partir del 0, incrementandose el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    La numeración partirá del origen de coordenadas.
    Args:
        pos_x (float): posición x sin escalar
        pos_y (float): posición y sin escalar
        cell_amount_x (int): Número de celdas en el eje x
        cell_amount_y (int): Número de celdas en el eje y
        max_position_x (float): Valor máximo de la posición x
        max_position_y (float): Valor máximo de la posición y
        use_caregorical_output (bool): Si se activa, devuelve un array de celdas numeradas a partir del 0, incrementandose el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    Returns:
        string: identificador de celda en el formato indicado
    '''

    # Creamos un listado para poder ahorrar código
    data = [
        [pos_x, cell_amount_x, max_position_x],
        [pos_y, cell_amount_y, max_position_y]
    ]
    result = []

    # Recorremos cada eje
    for axeData in data:
        # Obtenemos el tamaño de cada celda
        cell_size = axeData[2] / axeData[1]
        # Obtenemos el identificador de celda
        cell = math.ceil(axeData[0] / cell_size)
        if not use_caregorical_output:
            cell = str(cell)
        # Añadimos el identificador a la lista
        result.append(cell)

    if not use_caregorical_output:
        # Devolvemos el resultado
        return ','.join(result)

    # Multiplicamos y por la cantidad de celdas de X y sumamos x
    return (result[0] + ((result[1] - 1) * cell_amount_x) - 1)


def gridList_to_posXY(grid_list: np.array, cell_amount_x: int, cell_amount_y: int, max_position_x: float = 20.660138018121128, max_position_y: float = 17.64103475472807, use_caregorical_input: bool = True):
    '''
    Transforma una lista de identificadores de celda al centro de la celda correspondiente
    Se enumeran las celdas como {x_i},{y_i}, siendo la primera celda la 1,1.
    Si se usa el input categórico, la entrada será un indice numérico partiendo del 0, incrementandose en el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    La numeración partirá del origen de coordenadas.
    Args:
        grid_list (np.array): lista de identificadores de celda en el formato indicado
        cell_amount_x (int): Número de celdas en el eje x
        cell_amount_y (int): Número de celdas en el eje y
        max_position_x (float): Valor máximo de la posición x
        max_position_y (float): Valor máximo de la posición y
        use_caregorical_input (bool): Si se activa, la entrada será un indice numérico partiendo del 0, incrementandose en el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    Returns:
        np.array: posiciones x, y sin escalar
    '''
    result = [grid_to_posXY(grid, cell_amount_x, cell_amount_y, max_position_x,
                            max_position_y, use_caregorical_input=use_caregorical_input) for grid in grid_list]

    return np.array(result)


def grid_to_posXY(cell, cell_amount_x: int, cell_amount_y: int, max_position_x: float = 20.660138018121128, max_position_y: float = 17.64103475472807, use_caregorical_input: bool = True):
    '''
    Transforma un identificador de celda al centro de la celda correspondiente
    Se enumeran las celdas como {x_i},{y_i}, siendo la primera celda la 1,1.
    Si se usa el input categórico, la entrada será un indice numérico partiendo del 0, incrementandose en el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    La numeración partirá del origen de coordenadas.
    Args:
        cell (string): identificador de celda en el formato indicado
        cell_amount_x (int): Número de celdas en el eje x
        cell_amount_y (int): Número de celdas en el eje y
        max_position_x (float): Valor máximo de la posición x
        max_position_y (float): Valor máximo de la posición y
        use_caregorical_input (bool): Si se activa, la entrada será un indice numérico partiendo del 0, incrementandose en el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    Returns:
        float: posición x
        float: posición y
    '''

    # Primero, si es categórico, lo convertimos al formato {x_i},{y_i}, siendo la primera celda la 1,1.
    if use_caregorical_input:
        cell = cell_to_gridposition(cell, cell_amount_x)

    # Dividimos cell en cell_x y cell_y
    cell_x, cell_y = cell.split(',')

    # Calculamos el tamaño de cada celda
    cell_size_x = max_position_x / cell_amount_x
    cell_size_y = max_position_y / cell_amount_y

    # Calculamos la posición x e y en el centro de su celda
    pos_x = (int(cell_x) * cell_size_x) - (cell_size_x / 2)
    pos_y = (int(cell_y) * cell_size_y) - (cell_size_y / 2)

    # Devolvemos
    return pos_x, pos_y


def posXYList_to_dinamic_grid(pos_list: np.array, zoom_level: int, cell_amount_x: int, cell_amount_y: int, max_position_x: float = 20.660138018121128, max_position_y: float = 17.64103475472807, use_caregorical_output: bool = True):
    '''
    Transforma una lista de posiciones x, y al identificador de celda correspondiente en una rejilla dinámica (zoom)
    Se enumeran las celdas como {x_i},{y_i}, siendo la primera celda la 1,1.
    Si se usa el output categórico, se devuelve un array de celdas numeradas a partir del 0, incrementandose el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    La numeración partirá del origen de coordenadas y se incrementará siempre hacia la x primero y luego hacia la y
    Args:
        pos_list (np.array): lista de posiciones x, y sin escalar
        zoom_level (int): Nivel de zoom máximo requerido
        cell_amount_x (int): Número de celdas en el eje x
        cell_amount_y (int): Número de celdas en el eje y
        max_position_x (float): Valor máximo de la posición x
        max_position_y (float): Valor máximo de la posición y
        use_caregorical_output (bool): Si se activa, devuelve un array de celdas numeradas a partir del 0, incrementandose el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    Returns:
        np.array: identificadores de celda en el formato indicado
    '''
    result = [posXY_to_dinamic_grid(pos_x=pos_x, pos_y=pos_y, zoom_level=zoom_level, cell_amount_x=cell_amount_x, cell_amount_y=cell_amount_y,
                                    max_position_x=max_position_x, max_position_y=max_position_y, use_caregorical_output=use_caregorical_output) for pos_x, pos_y in pos_list]

    return np.array(result)


def posXY_to_dinamic_grid(pos_x: float, pos_y: float, zoom_level: int,  cell_amount_x: int, cell_amount_y: int, max_position_x: float = 20.660138018121128, max_position_y: float = 17.64103475472807, origin_x: float = 0, origin_y: float = 0, use_caregorical_output: bool = True):
    '''
    Transforma una posición x, y al identificador de celda correspondiente en una rejilla dinámica (zoom)
    Se enumeran las celdas como {x_i},{y_i}, siendo la primera celda la 1,1.
    Si se usa el output categórico, se devuelve un array de celdas numeradas a partir del 0, incrementandose el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    La numeración partirá del origen de coordenadas.
    Args:
        zoom_level (int): Nivel de zoom máximo requerido
        pos_x (float): posición x sin escalar
        pos_y (float): posición y sin escalar
        cell_amount_x (int): Número de celdas en el eje x
        cell_amount_y (int): Número de celdas en el eje y
        max_position_x (float): Valor máximo de la posición x
        max_position_y (float): Valor máximo de la posición y
        origin_x (float): Posición x del origen de coordenadas
        origin_y (float): Posición y del origen de coordenadas
        use_caregorical_output (bool): Si se activa, devuelve un array de celdas numeradas a partir del 0, incrementandose el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    Returns:
        np.array: identificadores de celda en el formato indicado
    '''
    result = []
    # Calculamos el tamaño de cada celda
    cell_size_x = max_position_x / cell_amount_x
    cell_size_y = max_position_y / cell_amount_y
    # Calculamos las coordenadas relativas al origen
    pos_x = pos_x - origin_x
    pos_y = pos_y - origin_y
    # Calculamos el identificador de celda
    grid_cell = posXY_to_grid(pos_x=pos_x, pos_y=pos_y, cell_amount_x=cell_amount_x, cell_amount_y=cell_amount_y,
                              max_position_x=max_position_x, max_position_y=max_position_y, use_caregorical_output=False)
    # Añadimos el valor al array de retorno
    grid_cell_final = grid_cell
    grid_data = grid_cell.split(',')
    if use_caregorical_output:
        grid_cell_final = gridposition_to_cell(
            int(grid_data[0]), int(grid_data[1]), cell_amount_x)
    result.append(grid_cell_final)

    # Bajamos en resolución si es necesario
    if zoom_level > 1:
        # Calculamos nuevos parámetros
        zoom_level -= 1
        max_position_x = cell_size_x
        max_position_y = cell_size_y
        origin_x = cell_size_x * (int(grid_data[0]) - 1)
        origin_y = cell_size_y * (int(grid_data[1]) - 1)
        # Recogemos el resultado de la siguiente resolución
        result += posXY_to_dinamic_grid(zoom_level=zoom_level, pos_x=pos_x, pos_y=pos_y, cell_amount_x=cell_amount_x, cell_amount_y=cell_amount_y,
                                        max_position_x=max_position_x, max_position_y=max_position_y, origin_x=origin_x, origin_y=origin_y, use_caregorical_output=use_caregorical_output)

    return result


def dinamic_gridList_to_posXY(cells_list: np.array, cell_amount_x: int, cell_amount_y: int, max_position_x: float = 20.660138018121128, max_position_y: float = 17.64103475472807, use_caregorical_input: bool = True):
    '''
    Transforma una lista de identificadores de celda al centro de la celda correspondiente en una rejilla dinámica (zoom)
    Se enumeran las celdas como {x_i},{y_i}, siendo la primera celda la 1,1.
    Si se usa el input categórico, la entrada será un indice numérico partiendo del 0, incrementandose en el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    La numeración partirá del origen de coordenadas.
    Args:
        cells_list (np.array): lista de identificadores de celda en el formato indicado
        cell_amount_x (int): Número de celdas en el eje x
        cell_amount_y (int): Número de celdas en el eje y
        max_position_x (float): Valor máximo de la posición x
        max_position_y (float): Valor máximo de la posición y
        use_caregorical_input (bool): Si se activa, la entrada será un indice numérico partiendo del 0, incrementandose en el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    Returns:
        np.array: listado de posiciones x, y sin escalar
    '''
    result = [dinamic_grid_to_posXY(cells=cells, cell_amount_x=cell_amount_x, cell_amount_y=cell_amount_y, max_position_x=max_position_x,
                                    max_position_y=max_position_y, use_caregorical_input=use_caregorical_input) for cells in cells_list]
    return np.array(result)


def dinamic_grid_to_posXY(cells: np.array, cell_amount_x: int, cell_amount_y: int, max_position_x: float = 20.660138018121128, max_position_y: float = 17.64103475472807, use_caregorical_input: bool = True):
    '''
    Transforma un identificador de celda correspondiente en una rejilla dinámica (zoom) a una posición x, y
    Se enumeran las celdas como {x_i},{y_i}, siendo la primera celda la 1,1.
    Si se usa el input categórico, la entrada será un indice numérico partiendo del 0, incrementandose en el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    La numeración partirá del origen de coordenadas.
    Args:
        cells (np.array): identificadores de celda en el formato indicado
        cell_amount_x (int): Número de celdas en el eje x
        cell_amount_y (int): Número de celdas en el eje y
        max_position_x (float): Valor máximo de la posición x
        max_position_y (float): Valor máximo de la posición y
        use_caregorical_input (bool): Si se activa, la entrada será un indice numérico partiendo del 0, incrementandose en el eje x primero. Ej 0 => 1,1, 1 => 2,1, 2 => 3,1, 3 => 1,2, 4 => 2,2, 5 => 3,2, 6 => 1,3, etc
    Returns:
        np.array: posición x, y sin escalar
    '''
    # Extraemos la posición de la primera celda
    cell = cells[0]
    cells = cells[1:]
    # Convertimos desde categorico si hace falta
    if use_caregorical_input:
        cell = cell_to_gridposition(cell, cell_amount_x)

    cell = cell.split(',')
    # Calculamos el tamaño de cada celda en esta dimension
    cell_size_x = max_position_x / cell_amount_x
    cell_size_y = max_position_y / cell_amount_y

    # Calculamos la posición de la celda en su punto origen
    pos_x = cell_size_x * (int(cell[0]) - 1)
    pos_y = cell_size_y * (int(cell[1]) - 1)

    # Si quedan celdas por procesar, bajamos en resolución
    if len(cells) > 0:
        # Calculamos nuevos parámetros
        max_position_x = cell_size_x
        max_position_y = cell_size_y
        # Recogemos el resultado de la siguiente resolución
        result = dinamic_grid_to_posXY(cells=cells, cell_amount_x=cell_amount_x, cell_amount_y=cell_amount_y,
                                       max_position_x=max_position_x, max_position_y=max_position_y, use_caregorical_input=use_caregorical_input)
        # Sumamos la posición de la celda en su punto origen
        pos_x += result[0]
        pos_y += result[1]
    else:
        # En el ultimo nivel sumamos la mitad de la celda para que el resultado sea en el centro de la misma
        pos_x += cell_size_x / 2
        pos_y += cell_size_y / 2

    # Devolvemos
    return [pos_x, pos_y]


def gridposition_to_cell(grid_x: int, grid_y: int, cell_amount_x: int):
    '''
    Transforma una posición en el grid (pareja de valor x,y siendo la primera 1,1) a identificador de celda (valor numérico partiendo del 0, incrementandose en el eje x primero)
    Args:
        grid_x (int): posición x en el grid
        grid_y (int): posición y en el grid
        cell_amount_x (int): Número de celdas en el eje x
    Returns:
        int: identificador de celda
    '''
    return (grid_x + ((grid_y - 1) * cell_amount_x) - 1)


def cell_to_gridposition(cell: int, cell_amount_x: int):
    '''
    Transforma un identificador de celda (valor numérico partiendo del 0, incrementandose en el eje x primero) a posición en el grid (pareja de valor x,y siendo la primera 1,1)
    Args:
        cell (int): identificador de celda
        cell_amount_x (int): Número de celdas en el eje x
    Returns:
        string: posición en el grid en el formato indicado (x,y) siendo la primera 1,1
    '''
    cell = int(cell)
    cell += 1
    cell_y = math.ceil(cell / cell_amount_x)
    cell_x = cell - ((cell_y - 1) * cell_amount_x)
    cell = str(cell_x)+','+str(cell_y)
    return cell


def groupDataForRnn(X: np.array, y: np.array, N: int, fill_empty_heads: bool = False, predict_next_output: bool = False):
    '''
    Agrupa los datos para un modelo RNN
    Args:
        X (np.array): Datos de entrada
        y (np.array): Datos de salida
        N (int): Tamaño de la secuencia
        fill_empty_heads (bool): Si se activa, se rellenan las primeras filas con la primera fila del dataset, evitando que se pierdan los primeros N valores
        predict_next_output (bool): Si se activa, se rellena la fila de salida con la siguiente fila del dataset
    Returns:
        np.array: Datos de entrada agrupados
        np.array: Datos de salida agrupados
    '''
    groupedX = []
    groupedy = []

    # Si se desea predecir, el ajuste hará que se i llegue hasta len(X)-1 y que Y acumule el valor de la siguiente fila (i+1)
    fix_to_predict = 0
    if (predict_next_output):
        fix_to_predict = 1

    if (fill_empty_heads):
        for i in range(0, len(X)-fix_to_predict):
            rowX = []
            # Si i es menor que N, no hemos llegado al tamaño de la secuencia
            if i < N:
                # Añadimos la primera fila N-i veces
                first_row = X[0]
                for j in range(0, N-i):
                    rowX.append(first_row)
                # Añadimos las filas desde 1 hasta i+1
                for j in range(1, i+1):
                    rowX.append(X[j])
            else:
                # Añadimos las filas desde i-N+1 hasta i+1 (el 0:i ya ha sido cubierto por el if previo)
                rowX = X[i-N+1:i+1]

            groupedX.append(rowX)
            groupedy.append(y[i+fix_to_predict])
    else:
        for i in range(N, len(X)):
            groupedX.append(X[i-N:i])
            groupedy.append(y[i-1+fix_to_predict])

    return np.array(groupedX), np.array(groupedy)

# endregion

# region Analisis y dibujado


def cross_val_score_multi_input(model: tf.keras.Model, X, y, cv, loss, optimizer, metrics, batch_size: int, epochs: int, verbose=0):
    '''
    Función que realiza cross validation de un modelo con múltiples entradas
    Args:
        model (keras.model): modelo a entrenar
        X (list): lista de arrays de entrada
        y (array): array de salida
        cv (sklearn.model_selection.KFold): objeto de validación cruzada
        batch_size (int): tamaño del batch
        epochs (int): número de épocas
        verbose (int, optional): nivel de verbosidad. Defaults to 0.
    Returns:
        np.array: lista de scores
    '''
    # Basado en https://stackoverflow.com/questions/59350224/crossvalidation-of-keras-model-with-multiply-inputs-with-scikit-learn
    cv_score = []
    # for i, (train, test) in enumerate(cv.split(X[0],y)):
    # Las dimensiones deben ser siempre las mismas. seguimos el ejemplo y usamso el primer índice
    for i, (train, test) in enumerate(cv.split(X, y)):
        # Clonamos el modelo para resetear los pesos
        model_clone = tf.keras.models.clone_model(model)
        model_clone.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

        # Preparamos la entrada X para train y test
        X_train = []
        X_test = []
        # for j in range(len(X)):
        if type(X) == pd.DataFrame:
            X_train.append(X.iloc[train])
            X_test.append(X.iloc[test])
        else:
            X_train.append(X[train])
            X_test.append(X[test])

        '''
        
        for j in range(len(X)):
            if type(X[j]) == pd.DataFrame:
                X_train.append(X[j].iloc[train])
                X_test.append(X[j].iloc[test])
            else:                
                X_train.append(X[j][train])
                X_test.append(X[j][test])
                '''

        # Ahora las Y
        if type(y) == pd.DataFrame:
            y_train = y.iloc[train]
            y_test = y.iloc[test]
        else:
            y_train = y[train]
            y_test = y[test]

        # Entrenamos el modelo y acumulamos el score
        print("Running Fold", i+1)
        model_clone.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size, verbose=verbose)
        result = model_clone.evaluate(X_test, y_test, verbose=verbose)
        # 0 es el loss siempre, si hay más de una métrica, se incrementa el índice
        cv_score.append(result[0])
        tf.keras.backend.clear_session()
    return np.array(cv_score)


def plot_learning_curves(hist, save_file: str = None, show_plot: bool = True):
    # Si no es diccionario bajamos al atributo history
    if not isinstance(hist, dict):
        hist = hist.history
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title('Curvas de aprendizaje')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Conjunto de entrenamiento',
               'Conjunto de validación'], loc='upper right')
    if save_file != None:
        plt.savefig(save_file)
    if show_plot:
        plt.show()
    else:
        plt.close()


def calculate_euclidean_distance(predictions: np.array, real_data: np.array) -> pd.DataFrame:
    """
    Calculate the Euclidean distance between predicted and real coordinates.

    Args:
        predictions (np.array): A numpy array of predicted coordinates, where each coordinate is a list [x, y].
        real_data (np.array): A numpy array of real coordinates, where each coordinate is a list [x, y].

    Returns:
        pandas.DataFrame: A DataFrame containing the predicted and real coordinates, their deviations, and the Euclidean distance.
    """

    # Obtenemos las coordenadas
    predicted_x = predictions[:, 0]
    predicted_y = predictions[:, 1]
    real_x = real_data[:, 0]
    real_y = real_data[:, 1]

    # Calculamos desviaciones
    deviation_x = np.abs(predicted_x - real_x)
    deviation_y = np.abs(predicted_y - real_y)

    # La distancia euclidiana
    euclidean_distance = np.sqrt(
        np.power(deviation_x, 2) + np.power(deviation_y, 2))

    # Componemos la salida
    return pd.DataFrame({
        'predicted_x': predicted_x,
        'predicted_y': predicted_y,
        'real_x': real_x,
        'real_y': real_y,
        'deviation_x': deviation_x,
        'deviation_y': deviation_y,
        'euclidean_distance': euclidean_distance
    })

# endregion

# region Herramientas generales


def save_model(model, model_file: str):
    '''
    Guarda un modelo en un fichero
    Args:
        model (keras.model): modelo a guardar
        model_file (str): ruta del fichero
    '''
    if os.path.exists(model_file):
        try:
            os.remove(model_file)
        except:
            shutil.rmtree(model_file)
    model.save(model_file)


def save_model_json(model, json_file: str, weights_file: Union[str, None] = None):
    """
    Save the model architecture as a JSON file and optionally save the model weights.
    Parameters:
        model (tf.keras.Model): The model to be saved.
        json_file (str): The path to save the model architecture JSON file.
        weights_file (str, optional): The path to save the model weights file. Defaults to None.
    Returns:
        None
    """
    # https://www.tensorflow.org/guide/keras/serialization_and_saving#model_serialization
    json_config = model.to_json()
    f = open(json_file, "a")
    f.write(json_config)
    f.close()

    if weights_file is not None:
        model.save_weights(weights_file)


def load_model_json(json_file: str, weights_file: Union[str, None] = None) -> tf.keras.models.Model:
    """
    Load a Keras model from a JSON file.
    Parameters:
        json_file (str): The path to the JSON file containing the model architecture.
        weights_file (Union[str, None], optional): The path to the file containing the model weights. Defaults to None.
    Returns:
        tf.keras.models.Model: The loaded Keras model.
    """
    with open(json_file, "r") as f:
        json_config = f.readlines()
    model = tf.keras.models.model_from_json(json_config)

    if weights_file is not None:
        model.load_weights(weights_file)

    return model


def save_history(history, history_file: str):
    '''
    Guarda el historial de un modelo en un fichero
    Args:
        history (keras.history): historial a guardar
        history_file (str): ruta del fichero
    '''
    # Si history es none, no guardamos nada
    if history == None:
        return
    if os.path.exists(history_file):
        try:
            os.remove(history_file)
        except:
            shutil.rmtree(history_file)
    with open(history_file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def load_history(history_file: str):
    '''
    Carga el historial de un modelo desde un fichero
    Args:
        history_file (str): ruta del fichero
    Returns:
        keras.history: historial cargado
    '''
    if os.path.exists(history_file):
        with open(history_file, 'rb') as file_pi:
            return pickle.load(file_pi)
    return None


def set_random_seed_value(seed: int = 42):
    '''
    Establece la semilla aleatoria de numpy y tensorflow
    Args:
        seed (int): semilla aleatoria
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Configuraciones adicionales para TensorFlow
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

# endregion

# region Deprecateds


def load_training_data(training_file: str, scaler_file: str = None, include_pos_z: bool = True, scale_y: bool = False, remove_not_full_rows: bool = False):
    # Deprecated, usar load_data
    return load_data(training_file, scaler_file, True, include_pos_z, scale_y, remove_not_full_rows)


def load_real_track_data(track_file: str, scaler_file: str = None, include_pos_z: bool = True, scale_y: bool = False, remove_not_full_rows: bool = False):
    # Deprecated, usar load_data
    return load_data(track_file, scaler_file, False, include_pos_z, scale_y, remove_not_full_rows)


def load_training_data_inverse(training_file: str, scaler_file: str, include_pos_z: bool = True, scale_y: bool = False, remove_not_full_rows: bool = False, separate_mac_and_pos: bool = False):
    # Deprecated, usar load_data_inverse
    return load_data_inverse(training_file, scaler_file, True, include_pos_z, scale_y, remove_not_full_rows, separate_mac_and_pos)


def load_real_track_data_inverse(track_file: str, scaler_file: str, include_pos_z: bool = True, scale_y: bool = False, remove_not_full_rows: bool = False, separate_mac_and_pos: bool = False):
    # Deprecated, usar load_data_inverse
    return load_data_inverse(track_file, scaler_file, False, include_pos_z, scale_y, remove_not_full_rows, separate_mac_and_pos)
# endregion
