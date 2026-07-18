import { User } from "lucide-react";

const STATUS_META = {
  online: {
    label: "Backend online",
    dotClass: "bg-green-400 shadow-[0_0_8px_rgba(74,222,128,0.45)]",
    textClass: "text-green-300",
  },
  "warming-up": {
    label: "Backend warming up",
    dotClass: "bg-yellow-400 animate-pulse shadow-[0_0_8px_rgba(250,204,21,0.4)]",
    textClass: "text-yellow-200",
  },
  unavailable: {
    label: "Backend unavailable",
    dotClass: "bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.4)]",
    textClass: "text-red-300",
  },
};

export default function Navbar({ userEmail, backendStatus = "warming-up" }) {
  const status = STATUS_META[backendStatus] || STATUS_META["warming-up"];

  return (
    <div className="min-h-14 flex items-center justify-between gap-4 px-6 py-2 border-b border-[#30363d] bg-[rgba(22,27,34,0.6)] backdrop-blur-xl transition-all duration-300">
      <div className="flex items-center gap-3 min-w-0">
        <h1 className="font-bold text-sm tracking-wider uppercase text-[#c9d1d9] whitespace-nowrap">
          Financial <span className="text-blue-400">Pragmatic</span> AI
        </h1>
        <div
          className={`flex items-center gap-2 border border-[#30363d] bg-[#0d1117]/60 px-2.5 py-1 rounded text-[10px] ${status.textClass}`}
          title={status.label}
        >
          <span className={`w-1.5 h-1.5 rounded-full ${status.dotClass}`} />
          <span className="hidden sm:inline">{status.label}</span>
        </div>
      </div>

      {userEmail && (
        <div className="flex items-center gap-2 text-[#8b949e] min-w-0">
          <User size={14} className="text-blue-500/70 shrink-0" />
          <span className="text-[11px] font-mono truncate max-w-48">
            {userEmail}
          </span>
        </div>
      )}
    </div>
  );
}
