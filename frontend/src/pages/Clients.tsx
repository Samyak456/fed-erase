import { useState } from 'react';
import { X, Eye, Trash2, Cpu, MemoryStick, Wifi, Zap, Circle } from 'lucide-react';
import { clients } from '../data/mockData';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

const statusColor: Record<string, string> = {
  active: '#22c55e',
  idle: '#f59e0b',
  disconnected: '#e05c5c',
};

const statusBg: Record<string, string> = {
  active: 'rgba(34,197,94,0.1)',
  idle: 'rgba(245,158,11,0.1)',
  disconnected: 'rgba(224,92,92,0.1)',
};

function ResourceBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full" style={{ background: '#eef3f9' }}>
        <div className="h-1.5 rounded-full" style={{ width: `${value}%`, background: color }} />
      </div>
      <span className="text-xs w-8 text-right" style={{ color: '#6a8caa', fontFamily: 'var(--font-mono)' }}>
        {value}%
      </span>
    </div>
  );
}

const mockHistory = [
  { round: 1, score: 0.14 }, { round: 4, score: 0.18 }, { round: 8, score: 0.21 },
  { round: 12, score: 0.23 }, { round: 16, score: 0.24 }, { round: 20, score: 0.24 },
];

interface ClientsPanelProps {
  client: typeof clients[0];
  onClose: () => void;
  onSelectUnlearning: () => void;
}

function ClientPanel({ client: c, onClose, onSelectUnlearning }: ClientsPanelProps) {
  return (
    <div
      className="fixed inset-0 z-40 flex justify-end"
      style={{ background: 'rgba(26,51,82,0.25)', backdropFilter: 'blur(2px)' }}
      onClick={onClose}
    >
      <div
        className="h-full w-full max-w-md overflow-y-auto animate-slide-in"
        style={{ background: '#fff', boxShadow: '-4px 0 32px rgba(26,51,82,0.12)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: 'var(--color-border)' }}>
          <div>
            <h2 className="font-display font-semibold text-base" style={{ color: '#1a3352' }}>{c.name}</h2>
            <div className="text-xs mt-0.5" style={{ color: '#6a8caa' }}>ID: {c.id} · {c.location}</div>
          </div>
          <button onClick={onClose} className="w-8 h-8 rounded-lg flex items-center justify-center cursor-pointer" style={{ background: '#eef3f9', color: '#6a8caa' }}>
            <X size={15} />
          </button>
        </div>

        <div className="p-6 space-y-5">
          {/* Status */}
          <div className="flex items-center gap-3">
            <span
              className="px-3 py-1 rounded-full text-xs font-semibold capitalize flex items-center gap-1.5"
              style={{ background: statusBg[c.status], color: statusColor[c.status] }}
            >
              <Circle size={7} fill={statusColor[c.status]} stroke="none" />
              {c.status}
            </span>
            <span className="text-xs" style={{ color: '#6a8caa' }}>Last seen: {c.lastSeen}</span>
          </div>

          {/* Resources */}
          <div className="rounded-xl p-4 space-y-3" style={{ background: '#eef3f9' }}>
            <div className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: '#6a8caa' }}>Resource Availability</div>
            <div className="space-y-2.5">
              <div className="flex items-center gap-2">
                <Cpu size={12} style={{ color: '#2d6a9f' }} />
                <span className="text-xs w-20" style={{ color: '#1a3352' }}>Compute</span>
                <ResourceBar value={c.compute} color="#2d6a9f" />
              </div>
              <div className="flex items-center gap-2">
                <MemoryStick size={12} style={{ color: '#5b9bd5' }} />
                <span className="text-xs w-20" style={{ color: '#1a3352' }}>Memory</span>
                <ResourceBar value={c.memory} color="#5b9bd5" />
              </div>
              <div className="flex items-center gap-2">
                <Wifi size={12} style={{ color: '#7ab3d8' }} />
                <span className="text-xs w-20" style={{ color: '#1a3352' }}>Bandwidth</span>
                <ResourceBar value={c.bandwidth} color="#7ab3d8" />
              </div>
              <div className="flex items-center gap-2">
                <Zap size={12} style={{ color: '#9ecbe8' }} />
                <span className="text-xs w-20" style={{ color: '#1a3352' }}>Energy</span>
                <ResourceBar value={c.energy} color="#9ecbe8" />
              </div>
            </div>
          </div>

          {/* Scores */}
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: 'Contribution Score', value: (c.contributionScore * 100).toFixed(1) + '%', color: '#2d6a9f' },
              { label: 'Influence Score', value: (c.influenceScore * 100).toFixed(1) + '%', color: '#5b9bd5' },
              { label: 'Model Updates', value: String(c.modelUpdates), color: '#7ab3d8' },
              { label: 'Data Points', value: c.dataPoints.toLocaleString(), color: '#9ecbe8' },
            ].map((item) => (
              <div key={item.label} className="rounded-xl p-3 border" style={{ borderColor: 'var(--color-border)', background: '#fff' }}>
                <div className="text-xs" style={{ color: '#6a8caa' }}>{item.label}</div>
                <div className="text-base font-bold mt-1 font-display" style={{ color: item.color }}>{item.value}</div>
              </div>
            ))}
          </div>

          {/* Contribution history chart */}
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: '#6a8caa' }}>Contribution History</div>
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={mockHistory} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eef3f9" />
                <XAxis dataKey="round" tick={{ fontSize: 9, fill: '#6a8caa' }} />
                <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 9, fill: '#6a8caa' }} />
                <Tooltip formatter={(v) => [`${(+(v as number) * 100).toFixed(1)}%`]} contentStyle={{ borderRadius: 8, border: '1px solid #d8e6f2', fontSize: 11 }} />
                <Line type="monotone" dataKey="score" stroke="#2d6a9f" strokeWidth={2} dot={{ fill: '#2d6a9f', r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Participation */}
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: '#6a8caa' }}>Participation</div>
            <div className="flex flex-wrap gap-1.5">
              {Array.from({ length: 20 }, (_, i) => (
                <div
                  key={i}
                  className="w-6 h-6 rounded-md text-[9px] font-mono flex items-center justify-center"
                  style={{
                    background: i < c.rounds ? '#ddeaf7' : '#eef3f9',
                    color: i < c.rounds ? '#2d6a9f' : '#c5d5e8',
                    fontWeight: i < c.rounds ? 600 : 400,
                  }}
                >
                  {i + 1}
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={onSelectUnlearning}
            className="w-full py-3 rounded-xl text-sm font-semibold text-white cursor-pointer flex items-center justify-center gap-2"
            style={{ background: '#2d6a9f' }}
          >
            <Trash2 size={15} />
            Select for Unlearning
          </button>
        </div>
      </div>
    </div>
  );
}

interface ClientsProps {
  onSelectForUnlearning: (clientId: string) => void;
}

export default function Clients({ onSelectForUnlearning }: ClientsProps) {
  const [selected, setSelected] = useState<typeof clients[0] | null>(null);

  return (
    <div className="space-y-4 animate-fade-in">
      <div
        className="bg-card rounded-2xl border overflow-hidden"
        style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
      >
        <div className="px-6 py-4 border-b flex items-center justify-between" style={{ borderColor: 'var(--color-border)' }}>
          <div>
            <h2 className="font-display font-semibold text-base" style={{ color: '#1a3352' }}>Registered Clients</h2>
            <p className="text-xs mt-0.5" style={{ color: '#6a8caa' }}>5 clients enrolled in the federated learning network</p>
          </div>
          <div className="flex items-center gap-2">
            {['active', 'idle', 'disconnected'].map((s) => (
              <span key={s} className="text-xs flex items-center gap-1.5 px-2 py-1 rounded-lg" style={{ background: statusBg[s], color: statusColor[s] }}>
                <Circle size={6} fill={statusColor[s]} stroke="none" /> {s}
              </span>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr style={{ background: '#f8fbfe', borderBottom: '1px solid var(--color-border)' }}>
                {['Client ID', 'Status', 'Compute', 'Memory', 'Bandwidth', 'Energy', 'Contribution', 'Influence', 'Actions'].map((h) => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide" style={{ color: '#6a8caa' }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {clients.map((c, idx) => (
                <tr
                  key={c.id}
                  className="border-b cursor-pointer"
                  style={{ borderColor: 'var(--color-border)', background: idx % 2 === 0 ? '#fff' : '#fafcfe' }}
                  onMouseEnter={(e) => ((e.currentTarget as HTMLTableRowElement).style.background = '#f0f6fc')}
                  onMouseLeave={(e) => ((e.currentTarget as HTMLTableRowElement).style.background = idx % 2 === 0 ? '#fff' : '#fafcfe')}
                >
                  <td className="px-4 py-3 font-mono text-xs font-semibold" style={{ color: '#2d6a9f' }}>{c.id}</td>
                  <td className="px-4 py-3">
                    <span
                      className="px-2 py-0.5 rounded-full text-xs font-medium capitalize flex items-center gap-1 w-fit"
                      style={{ background: statusBg[c.status], color: statusColor[c.status] }}
                    >
                      <Circle size={6} fill={statusColor[c.status]} stroke="none" />
                      {c.status}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="w-20">
                      <ResourceBar value={c.compute} color="#2d6a9f" />
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="w-20">
                      <ResourceBar value={c.memory} color="#5b9bd5" />
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="w-20">
                      <ResourceBar value={c.bandwidth} color="#7ab3d8" />
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="w-20">
                      <ResourceBar value={c.energy} color="#9ecbe8" />
                    </div>
                  </td>
                  <td className="px-4 py-3 font-mono text-xs" style={{ color: '#1a3352' }}>
                    {(c.contributionScore * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3 font-mono text-xs" style={{ color: '#1a3352' }}>
                    {(c.influenceScore * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setSelected(c)}
                        className="px-2 py-1 rounded-lg text-xs font-medium cursor-pointer flex items-center gap-1"
                        style={{ background: '#ddeaf7', color: '#2d6a9f' }}
                      >
                        <Eye size={11} /> View
                      </button>
                      <button
                        onClick={() => onSelectForUnlearning(c.id)}
                        className="px-2 py-1 rounded-lg text-xs font-medium cursor-pointer flex items-center gap-1"
                        style={{ background: 'rgba(224,92,92,0.1)', color: '#e05c5c' }}
                      >
                        <Trash2 size={11} /> Unlearn
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {selected && (
        <ClientPanel
          client={selected}
          onClose={() => setSelected(null)}
          onSelectUnlearning={() => {
            onSelectForUnlearning(selected.id);
            setSelected(null);
          }}
        />
      )}
    </div>
  );
}
