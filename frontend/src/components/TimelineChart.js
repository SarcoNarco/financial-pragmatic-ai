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
  EXPANSION: 1,
  COST_PRESSURE: -1,
  STRATEGIC_PROBING: -0.5,
  GENERAL_UPDATE: 0,
};

function smoothData(data, windowSize = 5) {
  return data.map((_, i, arr) => {
    const start = Math.max(0, i - windowSize);
    const end = Math.min(arr.length, i + windowSize);
    const subset = arr.slice(start, end);
    return subset.reduce((a, b) => a + b, 0) / subset.length;
  });
}

export default function TimelineChart({ segments }) {
  const labels = segments.map((_, i) => `Step ${i + 1}`);
  const signalData = segments.map((s) => intentWeights[s.intent] ?? 0);
  const smoothedData = smoothData(signalData);

  const data = {
    labels,
    datasets: [
      {
        label: "Conversation Flow",
        data: smoothedData,
        fill: true,
        tension: 0.4,
        borderWidth: 2,
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
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#252526",
        titleColor: "#d4d4d4",
        bodyColor: "#d4d4d4",
        borderColor: "#333",
        borderWidth: 1,
        callbacks: {
          title: (items) => {
            const index = items[0]?.dataIndex ?? 0;
            return `Step ${index + 1}`;
          },
          label: (item) => {
            const index = item.dataIndex;
            const segment = segments[index];
            const intent = segment?.intent || "GENERAL_UPDATE";
            return `Intent: ${intent}`;
          },
          afterLabel: (item) => {
            const index = item.dataIndex;
            const segment = segments[index];
            const text = segment?.text || "";
            return `Text: ${text}`;
          },
        },
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
          stepSize: 0.5,
          callback: (value) => {
            if (value === 1) return "Growth";
            if (value === -1) return "Risk";
            if (value === -0.5) return "Probe";
            return "Neutral";
          },
        },
        min: -1,
        max: 1,
        grid: { color: "#333" },
      },
    },
  };

  return <Line data={data} options={options} />;
}
