import express, { Router, Express } from 'express';
import IndexController from '../controllers';

const router = Router();
const indexController = new IndexController();

export function setRoutes(app: Express) {
    app.use(express.json());
    app.use('/', router);
    router.get('/', indexController.getIndex.bind(indexController));
    router.post('/predict', indexController.predict.bind(indexController)); // <-- This line is required!
}