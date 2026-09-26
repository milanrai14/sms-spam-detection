import { useState } from "react";

interface PredictionResponse {
  prediction: number;
  result: string;
}

function SpamDetector() {
  const [message, setMessage] = useState<string>("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const checkSpam = async (): Promise<void> => {
    if (!message.trim()) {
      setError("Please enter a message.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to connect to the API");
      }

      const data: PredictionResponse = await response.json();

      setResult(data);
    } catch (error) {
      console.error(error);
      setError("Unable to connect to the backend API.");
    } finally {
      setLoading(false);
    }
  };

  const clearMessage = (): void => {
    setMessage("");
    setResult(null);
    setError("");
  };

  return (
    <div className="spam-detector">
      <div className="header">
        <h1>SMS Spam Detector</h1>

        <p>
          Enter an SMS message to determine whether it is spam or not.
        </p>
      </div>

      <div className="input-section">
        <label htmlFor="message">SMS Message</label>

        <textarea
          id="message"
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          placeholder="Enter your SMS message here..."
          rows={7}
        />
      </div>

      <div className="button-group">
        <button
          className="check-button"
          onClick={checkSpam}
          disabled={loading}
        >
          {loading ? "Checking..." : "Check Message"}
        </button>

        <button
          className="clear-button"
          onClick={clearMessage}
          disabled={loading}
        >
          Clear
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <div
          className={`result ${
            result.prediction === 1 ? "spam" : "not-spam"
          }`}
        >
          <span className="result-label">Prediction</span>

          <h2>{result.result}</h2>

          <p>
            {result.prediction === 1
              ? "This message appears to be a spam message."
              : "This message appears to be a legitimate message."}
          </p>
        </div>
      )}
    </div>
  );
}

export default SpamDetector;