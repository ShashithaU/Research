import { Request, Response } from 'express';
import fetch from 'node-fetch'; // npm install node-fetch

class IndexController {
    getIndex(req: Request, res: Response) {
        res.send('Welcome to the backend API!');
    }

    async predict(req: Request, res: Response) {
        const { features } = req.body;
        if (!Array.isArray(features) || features.length !== 11) {
            return res.status(400).json({ error: 'features must be an array of 11 numbers' });
        }
        try {
            console.log('Received features:', features);
            const pyRes = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features })
            });
            const data = await pyRes.json();
            console.log('Prediction response:', data);
            res.json(data);
        } catch (err) {
            res.status(500).json({ error: 'Failed to get prediction from Python model' });
        }
    }
}

export default IndexController;