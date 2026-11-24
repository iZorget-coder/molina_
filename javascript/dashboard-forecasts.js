import { fetchLatestPredictions, formatDay, formatDate, weekdayLong } from './predictions.js';
import { renderDashboardAiUsage } from './ai-usage.js';

export async function loadForecastHeader() {
    renderDashboardAiUsage();

    const title = document.getElementById('forecast-title');
    const sub = document.getElementById('forecast-sub');
    if (!title || !sub) return;

  try {
    const data = await fetchLatestPredictions();
    if (!title.isConnected) return; 
    if (!data) {
      title.textContent = 'No forecasts yet';
      sub.textContent = 'Check back after the next daily run.';
      return;
    }

    title.textContent = `Forecasts for ${formatDay(data.forecastDay)}`;
    sub.textContent = data.closeDate
      ? `Based on the ${weekdayLong(data.closeDate)} close, ${formatDate(data.closeDate)}`
      : '';
  } catch (err) {
    console.error('Could not load forecasts', err);
    if (!title.isConnected) return;
    title.textContent = 'Forecasts unavailable';
    sub.textContent = 'Could not load the latest forecasts.';
  }
}