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

export default function TimelineChart({ segments }) {
  const labels = segments.map((_, i) => `Step ${i}`);
  const dataPoints = segments.map((s) => intentWeights[s.intent] ?? 0);

  const data = {
    labels,
    datasets: [
      {
        label: "Conversation Flow",
        data: dataPoints,
        tension: 0.4,
        borderColor: "#569cd6",
        backgroundColor: "rgba(86, 156, 214, 0.25)",
        pointBackgroundColor: "#9cdcfe",
        pointBorderColor: "#569cd6",
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
