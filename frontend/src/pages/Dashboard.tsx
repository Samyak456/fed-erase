import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Cpu, Users, RefreshCw, Trash2, ArrowUpRight, Circle } from 'lucide-react';
import { accuracyHistory, contributionData, influenceData, resourceUsage, clients } from '../data/mockData';

interface DashboardProps {
  onStartUnlearning: () => void;
}

function MetricCard({
  label,
  value,
  sub,
  icon: Icon,
  color,
}: {
  label: string;
  value: string;
  sub: string;
  icon: React.ComponentType<{ size?: number }>;
  color: string;
}) {
  return (
    <div
      className="bg-card rounded-2xl p-5 border flex flex-col gap-3 animate-fade-in"
      style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
    >
      <div className="flex items-start justify-between">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center"
          style={{ background: color + '18', color }}
        >
          <Icon size={18} />
        </div>
        <span className="text-xs font-medium flex items-center gap-0.5" style={{ color: '#22c55e' }}>
          <ArrowUpRight size={12} /> Live
        </span>
      </div>
      <div>
        <div className="text-2xl font-display font-bold" style={{ color: '#1a3352' }}>{value}</div>
        <div className="text-sm mt-0.5 font-medium" style={{ color: '#6a8caa' }}>{label}</div>
        <div className="text-xs mt-1" style={{ color: '#a0bdd0' }}>{sub}</div>
      </div>
    </div>
  );
}

const clientColors: Record<string, string> = {
  'C-01': '#5b9bd5',
  'C-02': '#7ab3d8',
  'C-03': '#9ecbe8',
  'C-04': '#2d6a9f',
  'C-05': '#b8d8f0',
};

const statusColor: Record<string, string> = {
  active: '#22c55e',
  idle: '#f59e0b',
  disconnected: '#e05c5c',
};

export default function Dashboard({ onStartUnlearning }: DashboardProps) {
  return (
    <div className="space-y-6">
      {/* Metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Global Model Accuracy" value="86.3%" sub="Round 20 of 20" icon={Cpu} color="#2d6a9f" />
        <MetricCard label="Active Clients" value="3 / 5" sub="2 idle or offline" icon={Users} color="#5b9bd5" />
        <MetricCard label="Current Round" value="Round 20" sub="Training complete" icon={RefreshCw} color="#7ab3d8" />
        <MetricCard label="Unlearning Status" value="Ready" sub="No pending requests" icon={Trash2} color="#9ecbe8" />
      </div>

      {/* Network visualization */}
      <div
        className="bg-card rounded-2xl border p-6 animate-fade-in"
        style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
      >
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="font-display font-semibold text-base" style={{ color: '#1a3352' }}>
              Federated Learning Network
            </h2>
            <p className="text-xs mt-0.5" style={{ color: '#6a8caa' }}>
              Client nodes connected to the global model aggregator
            </p>
          </div>
          <button
            onClick={onStartUnlearning}
            className="px-4 py-2 rounded-xl text-sm font-semibold text-white cursor-pointer"
            style={{ background: '#2d6a9f' }}
            onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.background = '#1a3352')}
            onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.background = '#2d6a9f')}
          >
            Start Unlearning
          </button>
        </div>

        {/* Network diagram */}
        <div className="relative flex items-center justify-center py-6">
          {/* SVG connections */}
          <svg className="absolute inset-0 w-full h-full" style={{ pointerEvents: 'none' }}>
            {/* Lines from center to each client */}
            {[0, 1, 2, 3, 4].map((i) => {
              const totalClients = 5;
              const angleStep = (2 * Math.PI) / totalClients;
              const startAngle = -Math.PI / 2;
              const angle = startAngle + i * angleStep;
              const radius = 38;
              const cx = 50;
              const cy = 50;
              const x = cx + radius * Math.cos(angle);
              const y = cy + radius * Math.sin(angle);
              return (
                <line
                  key={i}
                  x1={`${cx}%`}
                  y1={`${cy}%`}
                  x2={`${x}%`}
                  y2={`${y}%`}
                  stroke={clients[i].status === 'active' ? '#7ab3d8' : '#d8e6f2'}
                  strokeWidth="1.5"
                  strokeDasharray={clients[i].status === 'active' ? '0' : '4 4'}
                />
              );
            })}
          </svg>

          {/* Global model center */}
          <div
            className="absolute rounded-2xl px-5 py-3 text-center border shadow-md"
            style={{
              left: '50%',
              top: '50%',
              transform: 'translate(-50%, -50%)',
              background: '#1a3352',
              borderColor: '#2d6a9f',
              zIndex: 2,
            }}
          >
            <div className="text-white font-display font-bold text-sm">Global Model</div>
            <div className="text-xs mt-0.5" style={{ color: '#7ab3d8', fontFamily: 'var(--font-mono)' }}>
              acc: 86.3%
            </div>
          </div>

          {/* Client nodes */}
          <div className="relative w-full" style={{ paddingTop: '60%' }}>
            {clients.map((c, i) => {
              const totalClients = clients.length;
              const angleStep = (2 * Math.PI) / totalClients;
              const startAngle = -Math.PI / 2;
              const angle = startAngle + i * angleStep;
              const radius = 38;
              const cx = 50;
              const cy = 50;
              const x = cx + radius * Math.cos(angle);
              const y = cy + radius * Math.sin(angle);
              return (
                <div
                  key={c.id}
                  className="absolute flex flex-col items-center"
                  style={{
                    left: `${x}%`,
                    top: `${y}%`,
                    transform: 'translate(-50%, -50%)',
                    zIndex: 3,
                  }}
                >
                  <div
                    className="w-14 h-14 rounded-2xl flex flex-col items-center justify-center border shadow-sm"
                    style={{
                      background: c.status === 'active' ? '#ddeaf7' : c.status === 'idle' ? '#fef3cd' : '#fde8e8',
                      borderColor: clientColors[c.id] + '60',
                    }}
                  >
                    <Circle
                      size={8}
                      fill={statusColor[c.status]}
                      stroke="none"
                      className={c.status === 'active' ? 'animate-pulse-soft' : ''}
                    />
                    <div className="text-xs font-bold mt-1" style={{ color: '#1a3352' }}>
                      {c.id}
                    </div>
                    <div className="text-[9px]" style={{ color: '#6a8caa', fontFamily: 'var(--font-mono)' }}>
                      {Math.round(c.contributionScore * 100)}%
                    </div>
                  </div>
                  <div className="text-[10px] mt-1 font-medium" style={{ color: '#6a8caa' }}>
                    {c.name}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Accuracy chart */}
        <div
          className="bg-card rounded-2xl border p-5 animate-fade-in"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
        >
          <h3 className="font-display font-semibold text-sm mb-4" style={{ color: '#1a3352' }}>
            Model Accuracy — Training Rounds
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={accuracyHistory} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef3f9" />
              <XAxis dataKey="round" tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <YAxis domain={[0.55, 0.95]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <Tooltip
                formatter={(v) => [`${(+(v as number) * 100).toFixed(1)}%`]}
                contentStyle={{ borderRadius: 8, border: '1px solid #d8e6f2', fontSize: 12 }}
              />
              <Line type="monotone" dataKey="accuracy" stroke="#2d6a9f" strokeWidth={2} dot={false} name="Accuracy" />
              <Line type="monotone" dataKey="target" stroke="#b8d8f0" strokeWidth={1} strokeDasharray="4 4" dot={false} name="Target" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Contribution chart */}
        <div
          className="bg-card rounded-2xl border p-5 animate-fade-in"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
        >
          <h3 className="font-display font-semibold text-sm mb-4" style={{ color: '#1a3352' }}>
            Client Contribution Distribution
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={contributionData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef3f9" />
              <XAxis dataKey="round" tick={{ fontSize: 10, fill: '#6a8caa' }} label={{ value: 'Round', position: 'insideBottom', offset: -2, fontSize: 10, fill: '#6a8caa' }} />
              <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <Tooltip
                formatter={(v) => [`${(+(v as number) * 100).toFixed(1)}%`]}
                contentStyle={{ borderRadius: 8, border: '1px solid #d8e6f2', fontSize: 12 }}
              />
              <Legend wrapperStyle={{ fontSize: 10, color: '#6a8caa' }} />
              <Bar dataKey="C01" fill="#2d6a9f" radius={[2, 2, 0, 0]} name="C-01" />
              <Bar dataKey="C02" fill="#5b9bd5" radius={[2, 2, 0, 0]} name="C-02" />
              <Bar dataKey="C03" fill="#7ab3d8" radius={[2, 2, 0, 0]} name="C-03" />
              <Bar dataKey="C04" fill="#9ecbe8" radius={[2, 2, 0, 0]} name="C-04" />
              <Bar dataKey="C05" fill="#b8d8f0" radius={[2, 2, 0, 0]} name="C-05" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Influence + Resource */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div
          className="bg-card rounded-2xl border p-5 animate-fade-in"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
        >
          <h3 className="font-display font-semibold text-sm mb-4" style={{ color: '#1a3352' }}>
            Client Influence vs Contribution
          </h3>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={influenceData} layout="vertical" margin={{ top: 4, right: 12, bottom: 0, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef3f9" horizontal={false} />
              <XAxis type="number" domain={[0, 0.45]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <YAxis type="category" dataKey="client" tick={{ fontSize: 10, fill: '#6a8caa' }} width={32} />
              <Tooltip formatter={(v) => [`${(+(v as number) * 100).toFixed(1)}%`]} contentStyle={{ borderRadius: 8, border: '1px solid #d8e6f2', fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Bar dataKey="influence" fill="#2d6a9f" radius={[0, 3, 3, 0]} name="Influence" />
              <Bar dataKey="contribution" fill="#9ecbe8" radius={[0, 3, 3, 0]} name="Contribution" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div
          className="bg-card rounded-2xl border p-5 animate-fade-in"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
        >
          <h3 className="font-display font-semibold text-sm mb-4" style={{ color: '#1a3352' }}>
            Resource Usage (24h)
          </h3>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={resourceUsage} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef3f9" />
              <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <Tooltip formatter={(v) => [`${v}%`]} contentStyle={{ borderRadius: 8, border: '1px solid #d8e6f2', fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Line type="monotone" dataKey="cpu" stroke="#2d6a9f" strokeWidth={2} dot={false} name="CPU" />
              <Line type="monotone" dataKey="mem" stroke="#7ab3d8" strokeWidth={2} dot={false} name="Memory" />
              <Line type="monotone" dataKey="bw" stroke="#9ecbe8" strokeWidth={2} dot={false} name="Bandwidth" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
