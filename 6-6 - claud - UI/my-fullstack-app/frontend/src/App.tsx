import React, { useState } from 'react';

const FEATURE_COUNT = 11;

function App() {
  const [features, setFeatures] = useState<number[]>(Array(FEATURE_COUNT).fill(0));
  const [result, setResult] = useState<null | {
    prediction: string;
    probability: string;
    details: string;
  }>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFeatureChange = (idx: number, value: string) => {
    const newFeatures = [...features];
    newFeatures[idx] = Number(value);
    setFeatures(newFeatures);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await fetch('http://localhost:3000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features }),
      });
      if (!res.ok) throw new Error('Prediction failed');
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError('Failed to get prediction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1 className="app-title">Loan Classifier Demo</h1>
      <form onSubmit={handleSubmit} className="prediction-form">
        <div className="features-grid">
          {features.map((val, idx) => (
            <div key={idx} className="feature-input">
              <label className="feature-label">
                Feature {idx + 1}:
                <input
                  type="number"
                  value={val}
                  onChange={e => handleFeatureChange(idx, e.target.value)}
                  className="feature-input-field"
                  required
                />
              </label>
            </div>
          ))}
        </div>
        <button type="submit" disabled={loading} className={`predict-button ${loading ? 'loading' : ''}`}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>
      {error && <p className="error-message">{error}</p>}
      {result && (
        <div className="result-container">
          <h2 className="result-title">Prediction Result</h2>
          <div className="result-item">
            <span className="result-label">Prediction:</span>
            <span className={`result-value prediction-badge ${result.prediction.toLowerCase() === 'approved' ? 'approved' : 'rejected'}`}>
              {result.prediction}
            </span>
          </div>
          <div className="result-item">
            <span className="result-label">Probability:</span>
            <span className="result-value">{result.probability}</span>
          </div>
          <div className="result-item">
            <span className="result-label">Details:</span>
            <span className="result-value result-details">{result.details}</span>
          </div>
        </div>
      )}
      
      <style>{`
        .app-container {
          max-width: 600px;
          margin: 2rem auto;
          padding: 2rem;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          background: white;
          border-radius: 12px;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }
        
        .app-title {
          text-align: center;
          color: #2c3e50;
          margin-bottom: 2rem;
          font-weight: 600;
        }
        
        .prediction-form {
          margin-bottom: 2rem;
        }
        
        .features-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          gap: 1rem;
          margin-bottom: 1.5rem;
        }
        
        .feature-input {
          display: flex;
          flex-direction: column;
        }
        
        .feature-label {
          font-size: 0.9rem;
          font-weight: 500;
          color: #555;
          margin-bottom: 0.5rem;
        }
        
        .feature-input-field {
          padding: 0.5rem;
          border: 2px solid #e1e5e9;
          border-radius: 6px;
          font-size: 1rem;
          transition: border-color 0.2s ease;
        }
        
        .feature-input-field:focus {
          outline: none;
          border-color: #3498db;
          box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        
        .predict-button {
          width: 100%;
          padding: 0.875rem 1.5rem;
          background: linear-gradient(135deg, #3498db, #2980b9);
          color: white;
          border: none;
          border-radius: 8px;
          font-size: 1rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        
        .predict-button:hover:not(:disabled) {
          background: linear-gradient(135deg, #2980b9, #21618c);
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
        }
        
        .predict-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
          transform: none;
        }
        
        .predict-button.loading {
          background: linear-gradient(135deg, #95a5a6, #7f8c8d);
        }
        
        .error-message {
          color: #e74c3c;
          background: #fdf2f2;
          padding: 0.75rem;
          border-radius: 6px;
          border-left: 4px solid #e74c3c;
          margin: 1rem 0;
        }
        
        .result-container {
          background: #f8f9fa;
          padding: 1.5rem;
          border-radius: 8px;
          border: 1px solid #e9ecef;
          margin-top: 1.5rem;
        }
        
        .result-title {
          color: #2c3e50;
          margin-bottom: 1rem;
          font-size: 1.25rem;
        }
        
        .result-container p {
          margin: 0.5rem 0;
          line-height: 1.5;
        }
        
        .result-container strong {
          color: #2c3e50;
        }
      `}</style>
    </div>
  );
}

export default App;