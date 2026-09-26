import { Cpu, MemoryStick, Wifi, Zap, ArrowDown, Target } from 'lucide-react';
import { clients } from '../data/mockData';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip,
} from 'recharts';

function ResourceGauge({ label, value, color, icon: Icon }: { label: string; value: number; color: string; icon: React.ComponentType<{ size?: number }> }) {
  return (
    <div className="flex flex-col items-center gap-2 p-4">
      <div className="w-12 h-12 rounded-2xl flex items-center justify-center" style={{ background: color + '18', color }}>
        <Icon size={20} />
      </div>
      <div className="text-center">
        <div className="relative w-16 h-16 mx-auto">
          <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
            <circle cx="18" cy="18" r="15.9" fill="none" stroke="#eef3f9" strokeWidth="3" />
            <circle
              cx="18" cy="18" r="15.9" fill="none"
              stroke={color} strokeWidth="3"
              strokeDasharray={`${value} ${100 - value}`}
              strokeLinecap="round"
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xs font-bold font-mono" style={{ color: '#1a3352' }}>{value}%</span>
          </div>
        </div>
        <div className="text-xs mt-1 font-medium" style={{ color: '#6a8caa' }}>{label}</div>
      </div>
    </div>
  );
}

const strategyByResource = (compute: number, memory: number, _bw: number) => {
  if (compute >= 80 && memory >= 75) return { strategy: 'Deep Unlearning', compression: 'Low', repairRounds: 3, checkpointFreq: 'Every step' };
  if (compute >= 50 && memory >= 45) return { strategy: 'Standard Unlearning', compression: 'Medium', repairRounds: 2, checkpointFreq: 'Every 2 steps' };
  return { strategy: 'Lightweight Unlearning', compression: 'High', repairRounds: 1, checkpointFreq: 'Every 3 steps' };
};

const strategyColors: Record<string, string> = {
  'Deep Unlearning': '#2d6a9f',
  'Standard Unlearning': '#5b9bd5',
  'Lightweight Unlearning': '#9ecbe8',
};

export default function ResourceAdaptation() {
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Intro banner */}
      <div
        className="rounded-2xl p-5 border"
        style={{ background: 'rgba(45,106,159,0.06)', borderColor: '#d8e6f2' }}
      >
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0" style={{ background: '#ddeaf7', color: '#2d6a9f' }}>
            <Target size={18} />
          </div>
          <div>
            <div className="font-display font-semibold text-sm" style={{ color: '#1a3352' }}>Adaptive Resource Policy</div>
            <p className="text-xs mt-1 leading-relaxed" style={{ color: '#6a8caa' }}>
              FedErase dynamically selects an unlearning strategy for each client based on its available compute, memory, bandwidth,
              and energy. High-resource clients receive a more thorough deep unlearning with low compression; resource-constrained
              clients receive a lightweight strategy with high compression and fewer repair rounds — ensuring the framework runs
              effectively across heterogeneous devices.
            </p>
          </div>
        </div>
      </div>

      {/* Per-client resource cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {clients.slice(0, 3).map((c) => {
          const strat = strategyByResource(c.compute, c.memory, c.bandwidth);
          const radarData = [
            { subject: 'CPU', A: c.compute },
            { subject: 'Memory', A: c.memory },
            { subject: 'Bandwidth', A: c.bandwidth },
            { subject: 'Energy', A: c.energy },
          ];
          return (
            <div
              key={c.id}
              className="bg-card rounded-2xl border p-5"
              style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
            >
              <div className="flex items-center justify-between mb-4">
                <div>
                  <div className="font-display font-semibold text-sm" style={{ color: '#1a3352' }}>{c.name}</div>
                  <div className="text-xs mt-0.5" style={{ color: '#6a8caa' }}>{c.id} · {c.location}</div>
                </div>
                <span
                  className="px-2 py-1 rounded-lg text-xs font-semibold"
                  style={{ background: strategyColors[strat.strategy] + '18', color: strategyColors[strat.strategy] }}
                >
                  {strat.strategy}
                </span>
              </div>

              {/* Gauges */}
              <div className="grid grid-cols-4 gap-1 mb-3">
                <ResourceGauge label="CPU" value={c.compute} color="#2d6a9f" icon={Cpu} />
                <ResourceGauge label="RAM" value={c.memory} color="#5b9bd5" icon={MemoryStick} />
                <ResourceGauge label="BW" value={c.bandwidth} color="#7ab3d8" icon={Wifi} />
                <ResourceGauge label="Pwr" value={c.energy} color="#9ecbe8" icon={Zap} />
              </div>

              {/* Strategy params */}
              <div className="rounded-xl p-3 space-y-1.5" style={{ background: '#f8fbfe' }}>
                {[
                  { label: 'Compression', value: strat.compression },
                  { label: 'Repair Rounds', value: strat.repairRounds },
                  { label: 'Checkpoint Freq', value: strat.checkpointFreq },
                ].map((r) => (
                  <div key={r.label} className="flex justify-between text-xs">
                    <span style={{ color: '#6a8caa' }}>{r.label}</span>
                    <span className="font-mono font-semibold" style={{ color: '#1a3352' }}>{r.value}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Decision flow */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div
          className="bg-card rounded-2xl border p-5"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
        >
          <h3 className="font-display font-semibold text-sm mb-5" style={{ color: '#1a3352' }}>Adaptive Strategy Decision Flow</h3>
          <div className="space-y-2">
            {[
              { label: 'Available Resources', desc: 'CPU, Memory, Bandwidth, Energy collected from each client', icon: Cpu, color: '#2d6a9f' },
              { label: 'Adaptive Policy Engine', desc: 'Thresholds evaluated: high (≥80%), medium (≥50%), low (<50%)', icon: Target, color: '#5b9bd5' },
              { label: 'Selected Strategy', desc: 'Deep / Standard / Lightweight unlearning with corresponding parameters', icon: Zap, color: '#7ab3d8' },
            ].map((step, i) => (
              <div key={i} className="flex flex-col items-center">
                <div
                  className="w-full rounded-xl p-4 border flex items-start gap-3"
                  style={{ borderColor: step.color + '40', background: step.color + '08' }}
                >
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0" style={{ background: step.color + '18', color: step.color }}>
                    <step.icon size={15} />
                  </div>
                  <div>
                    <div className="text-sm font-semibold" style={{ color: '#1a3352' }}>{step.label}</div>
                    <div className="text-xs mt-0.5" style={{ color: '#6a8caa' }}>{step.desc}</div>
                  </div>
                </div>
                {i < 2 && (
                  <div className="flex flex-col items-center py-1.5">
                    <div className="w-px h-4" style={{ background: '#d8e6f2' }} />
                    <ArrowDown size={12} style={{ color: '#a0bdd0' }} />
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Strategy table */}
          <div className="mt-5 rounded-xl overflow-hidden border" style={{ borderColor: 'var(--color-border)' }}>
            <table className="w-full text-xs">
              <thead style={{ background: '#f0f6fc' }}>
                <tr>
                  {['Resources', 'Strategy', 'Compression', 'Repair Rounds'].map((h) => (
                    <th key={h} className="px-3 py-2 text-left font-semibold" style={{ color: '#6a8caa' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  ['High (≥80%)', 'Deep Unlearning', 'Low', '3'],
                  ['Medium (≥50%)', 'Standard', 'Medium', '2'],
                  ['Low (<50%)', 'Lightweight', 'High', '1'],
                ].map((row, i) => (
                  <tr key={i} className="border-t" style={{ borderColor: '#eef3f9' }}>
                    {row.map((cell, j) => (
                      <td key={j} className="px-3 py-2 font-mono" style={{ color: j === 0 ? '#6a8caa' : '#1a3352' }}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Radar chart */}
        <div
          className="bg-card rounded-2xl border p-5"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
        >
          <h3 className="font-display font-semibold text-sm mb-4" style={{ color: '#1a3352' }}>Client Resource Profile — C-04 (High Resources)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <RadarChart data={[
              { subject: 'Compute', val: 91 },
              { subject: 'Memory', val: 88 },
              { subject: 'Bandwidth', val: 94 },
              { subject: 'Energy', val: 89 },
            ]}>
              <PolarGrid stroke="#eef3f9" />
              <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: '#6a8caa' }} />
              <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 9, fill: '#c5d5e8' }} />
              <Radar dataKey="val" stroke="#2d6a9f" fill="#2d6a9f" fillOpacity={0.15} strokeWidth={2} />
              <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #d8e6f2', fontSize: 11 }} formatter={(v) => [`${v}%`]} />
            </RadarChart>
          </ResponsiveContainer>
          <div className="mt-3 rounded-xl p-3 text-center" style={{ background: '#ddeaf7' }}>
            <span className="text-xs font-semibold" style={{ color: '#2d6a9f' }}>→ Assigned: Deep Unlearning · Low Compression · 3 Repair Rounds</span>
          </div>
        </div>
      </div>

      {/* All clients summary table */}
      <div
        className="bg-card rounded-2xl border overflow-hidden"
        style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
      >
        <div className="px-6 py-4 border-b" style={{ borderColor: 'var(--color-border)' }}>
          <h3 className="font-display font-semibold text-sm" style={{ color: '#1a3352' }}>All Clients — Adaptive Strategy Assignment</h3>
        </div>
        <table className="w-full text-sm">
          <thead style={{ background: '#f8fbfe' }}>
            <tr>
              {['Client', 'CPU', 'Memory', 'Bandwidth', 'Energy', 'Strategy', 'Compression', 'Repair Rounds'].map((h) => (
                <th key={h} className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide" style={{ color: '#6a8caa' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {clients.map((c, idx) => {
              const s = strategyByResource(c.compute, c.memory, c.bandwidth);
              return (
                <tr key={c.id} className="border-t" style={{ borderColor: '#eef3f9', background: idx % 2 === 0 ? '#fff' : '#fafcfe' }}>
                  <td className="px-4 py-3 font-mono text-xs font-semibold" style={{ color: '#2d6a9f' }}>{c.id}</td>
                  <td className="px-4 py-3 font-mono text-xs" style={{ color: '#1a3352' }}>{c.compute}%</td>
                  <td className="px-4 py-3 font-mono text-xs" style={{ color: '#1a3352' }}>{c.memory}%</td>
                  <td className="px-4 py-3 font-mono text-xs" style={{ color: '#1a3352' }}>{c.bandwidth}%</td>
                  <td className="px-4 py-3 font-mono text-xs" style={{ color: '#1a3352' }}>{c.energy}%</td>
                  <td className="px-4 py-3">
                    <span className="px-2 py-0.5 rounded-lg text-xs font-medium" style={{ background: strategyColors[s.strategy] + '18', color: strategyColors[s.strategy] }}>
                      {s.strategy}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-mono text-xs" style={{ color: '#1a3352' }}>{s.compression}</td>
                  <td className="px-4 py-3 font-mono text-xs" style={{ color: '#1a3352' }}>{s.repairRounds}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
