import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend
);

const intentWeights = {
  EXPANSION: 2,
  COST_PRESSURE: -2,
  STRATEGIC_PROBING: -1,
  GENERAL_UPDATE: 0,
};

function movingAverage(points, windowSize = 2) {
  return points.map((_, index) => {
    const start = Math.max(0, index - windowSize);
    const end = Math.min(points.length - 1, index + windowSize);
    const slice = points.slice(start, end + 1);
    const total = slice.reduce((sum, value) => sum + value, 0);
    return total / slice.length;
  });
}

export default function TimelineChart({ segments }) {
  const labels = segments.map((_, i) => `Step ${i}`);
  const rawPoints = segments.map((s) => intentWeights[s.intent] ?? 0);
  const dataPoints = movingAverage(rawPoints, 2);

  const data = {
    labels,
    datasets: [
      {
        label: "Conversation Flow",
        data: dataPoints,
        fill: true,
        tension: 0.4,
        borderColor: "#569cd6",
        backgroundColor: (context) => {
          const chart = context.chart;
          const { ctx, chartArea } = chart;
          if (!chartArea) {
            return "rgba(86, 156, 214, 0.15)";
          }
          const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
          gradient.addColorStop(0, "rgba(86, 156, 214, 0.40)");
          gradient.addColorStop(1, "rgba(86, 156, 214, 0.03)");
          return gradient;
        },
        pointRadius: 0,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#252526",
        titleColor: "#d4d4d4",
        bodyColor: "#d4d4d4",
        borderColor: "#333",
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        ticks: { color: "#d4d4d4" },
        grid: { color: "#333" },
      },
      y: {
        ticks: {
          color: "#d4d4d4",
          stepSize: 1,
          callback: (value) => {
            if (value === 2) return "Growth";
            if (value === -2) return "Risk";
            if (value === -1) return "Uncertainty";
            return "Neutral";
          },
        },
        min: -2,
        max: 2,
        grid: { color: "#333" },
      },
    },
  };

  return <Line data={data} options={options} />;
}
