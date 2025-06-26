import express from 'express';
import cors from 'cors';
import { setRoutes } from './routes';

const app = express();
app.use(cors()); // <-- Add this line

setRoutes(app);

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});