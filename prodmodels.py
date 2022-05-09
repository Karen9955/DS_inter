import os
import functools as ft

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

import numpy as np
import pickle
import pandas as pd
import torch


class MainRecommender():
    """Верхнеуровневый класс выбора алгоритма рекомендательной системы для исполнения."""
    
    def __init__(self):
        
        path_to_clients = ft.reduce(os.path.join, ['data', 'rfm_client_hist_context_target_product.csv'])
        self.clients = pd.read_csv(path_to_clients)['client_idx'].unique().tolist()
        
        self.anon_model = AnonymousXGBRecommender()
        self.hist_model = XGBRecommender()
        
    def predict_sample(self, user_id, basket, store_idx, time, slot_num):
        
        if (user_id in self.clients):
            return self.hist_model.predict_sample(user_id, basket, store_idx, time, slot_num)
        else:
            return self.anon_model.predict_sample(user_id, basket, store_idx, time, slot_num)   
            
            
class AnonymousXGBRecommender():
    
    """Класс-обертка для анонимного градиентного бустинга."""
    
    def __init__(self):
        """Инициализация класса. Загрузка состояний моделей и датасетов."""
        
        self.svd_model = self.load_model('svd_model_anon')
        self.xgb_model = self.load_model('xgb_model_anon')
        
        self.products = pd.read_csv(ft.reduce(os.path.join, ['data', 'processed', 'products.csv']), index_col=0)
        self.prod_num = self.products.shape[0]
        self.products = self.products.sort_index()
        self.products['target_product'] = range(self.prod_num) 
        #self.products.drop('triangle_cat', axis=1, inplace=True)
        
        #path_to_dataset = ft.reduce(os.path.join, [root, 'data', 'client_hist_context_target_product.csv'])
        #dataset_header = pd.read_csv(path_to_dataset, index_col=0, nrows=0)
        #cols = ['h_' + str(x) for x in range(self.prod_num)] + ['is_new_client', 'age', 'gender', 'target']
        #dataset_header.drop(cols, inplace=True, axis=1)
        #dataset_header = dataset_header.columns.tolist()
        #self.dataset_header = dataset_header
        
        #self.trianglefilter = catfilter.TriangleCatFilter()
        
    def load_model(self, name):
        
        path = ft.reduce(os.path.join, ['models','pkl' ,name+'.pkl'])
        with open(path, 'rb') as handle:
            return pickle.load(handle)         
    
    def predict_sample(self, user_id, basket, store_idx, time, slot_num):
        """Функция возвращает рекомендованные к покупке товары и их рейтинг.
        Рекомендации осуществляются на основе id клиента и его текущей корзины. """
        
        basket_header = ['b_' + str(x) for x in range(self.prod_num)]
        
        # Приведение списка товаров в корзине к вектору-индикатору наличия товара в корзине
        basket_ = np.full((self.prod_num), False, dtype=bool)
        basket_[basket] = True
        
        # Сбор контекста
        user = pd.Series(basket_, index=basket_header)
        time = pd.to_datetime(time)
        user['store_idx'] = store_idx
        user['product_count'] = np.sum(basket_)
        user['hour'] = time.hour
        user['dayofweek'] = time.dayofweek
        user['day'] = time.day
        user['month'] = time.month
        user['year'] = time.year
        
        # Подготовка датасета вида клиент(контекст)-товары для скоринга товаров
        user_df = pd.concat([user] * self.prod_num, axis=1).transpose()
        user_df = pd.concat([user_df.reset_index(drop=True), self.products], axis=1)
        
        # Фильтрация по правилу треугольника
        #cats_to_rec = self.trianglefilter.filter_products(basket)
        #prods_to_rec = user_df[~user_df.triangle_cat.isin(cats_to_rec)].index.values

        #user_df = user_df[self.dataset_header]
        
        # Расчет скора SVD_KNN
        user_df['knn_score'] = cosine_similarity(self.svd_model.X_train_svd, self.svd_model.X_train_svd[basket].sum(0).reshape(1, -1))

        y_pred = self.xgb_model.predict(user_df.values)
        
        # Ранжируем, отбираем топ
        y_pred[basket] = -np.inf
        #y_pred[prods_to_rec] = -np.inf
        recs = np.argpartition(y_pred, -slot_num, axis=0)[-slot_num:]
        recs = list(recs[np.argsort(y_pred[recs])])
        recs.reverse()
        #scores = list(y_pred[recs])
        
        return recs#(recs, scores)

    
class XGBRecommender():
    """Класс-обертка для градиентного бустинга."""
    
    def __init__(self):
        """Инициализация класса. Загрузка состояний моделей и датасетов."""
             
        self.svd_model = self.load_model('svd_model')
        self.xgb_model = self.load_model('xgb_model')
        
        self.dataset = pd.read_csv(ft.reduce(os.path.join, ['data', 'rfm_client_hist_context_target_product.csv']), index_col=0)
        self.dataset = self.dataset[self.dataset.target == 1]
        self.dataset.drop(['transaction_datetime', 'transaction_idx', 'is_new_client'], axis=1, inplace=True)
        
        self.products = pd.read_csv(ft.reduce(os.path.join, ['data', 'processed', 'products.csv']), index_col=0)
        self.prod_num = self.products.shape[0]
        self.products = self.products.sort_index()
        self.products['target_product'] = range(self.prod_num) 
        #self.products.drop('triangle_cat', axis=1, inplace=True)
        
        #self.trianglefilter = catfilter.TriangleCatFilter()
        
        
    def load_model(self, name):
        
        path = ft.reduce(os.path.join, ['models','pkl' ,name+'.pkl'])
        with open(path, 'rb') as handle:
            return pickle.load(handle)         
    
    def predict_sample(self, user_id, basket, store_idx, time, slot_num):
        """Функция возвращает рекомендованные к покупке товары и их рейтинг.
        Рекомендации осуществляются на основе id клиента и его текущей корзины. """
        
        basket_header = ['b_' + str(x) for x in range(self.prod_num)]
        history_header = ['h_' + str(x) for x in range(self.prod_num)]

        # Формируем историю покупок клиента
        user = self.dataset.loc[user_id].copy()
        user.loc[history_header] = user.loc[basket_header].values | user.loc[history_header].values
        user.loc['h_'+str(user.target_product)] = True

        # Приведение списка товаров в корзине к вектору-индикатору наличия товара в корзине
        basket_ = np.full((self.prod_num), False, dtype=bool)
        basket_[basket] = True
        user.loc[basket_header] = basket_
        user.drop(self.products.columns, inplace=True)
        
        # Сбор контекста
        time = pd.to_datetime(time)
        user['store_idx'] = store_idx
        user['product_count'] = np.sum(basket_)
        user['hour'] = time.hour
        user['dayofweek'] = time.dayofweek
        user['day'] = time.day
        user['month'] = time.month
        user['year'] = time.year
        
        # Подготовка датасета вида клиент(контекст)-товары для скоринга товаров
        user_df = pd.concat([user] * self.prod_num, axis=1).transpose().drop('target', axis=1)
        user_df = pd.concat([user_df.reset_index(drop=True), self.products], axis=1)
        
        # Фильтрация по правилу треугольника
        #cats_to_rec = self.trianglefilter.filter_products(basket)
        #prods_to_rec = user_df[~user_df.triangle_cat.isin(cats_to_rec)].index.values
        
        user_df = user_df[self.dataset.drop('target', axis=1).columns]
        
        # Расчет скора SVD_KNN
        user_df['knn_score'] = cosine_similarity(self.svd_model.X_train_svd, self.svd_model.X_train_svd[basket].sum(0).reshape(1, -1))

        y_pred = self.xgb_model.predict(user_df.values)
        
        # Ранжируем, отбираем топ        
        y_pred[basket] = -np.inf
        #y_pred[prods_to_rec] = -np.inf
        recs = np.argpartition(y_pred, -slot_num, axis=0)[-slot_num:]
        recs = list(recs[np.argsort(y_pred[recs])])
        recs.reverse()
        #scores = list(y_pred[recs])
        
        return recs#(recs, scores)
