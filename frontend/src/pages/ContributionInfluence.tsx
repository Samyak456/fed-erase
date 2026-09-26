import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { TrendingUp, Database, Layers, Maximize2 } from 'lucide-react';
import { clients, contributionData, influenceData } from '../data/mockData';

const colors = ['#2d6a9f', '#5b9bd5', '#7ab3d8', '#9ecbe8', '#b8d8f0'];

const updateSizes = [
  { round: 1, C01: 2.1, C02: 1.8, C03: 1.4, C04: 2.5, C05: 0.9 },
  { round: 5, C01: 2.3, C02: 1.9, C03: 1.5, C04: 2.7, C05: 0.8 },
  { round: 10, C01: 2.2, C02: 1.8, C03: 1.4, C04: 2.6, C05: 0.7 },
  { round: 15, C01: 2.1, C02: 1.7, C03: 1.3, C04: 2.5, C05: 0.7 },
  { round: 20, C01: 2.0, C02: 1.7, C03: 1.3, C04: 2.5, C05: 0.6 },
];

export default function ContributionInfluence() {
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Target client highlight */}
      <div
        className="rounded-2xl p-6 border relative overflow-hidden"
        style={{
          background: 'linear-gradient(135deg, #1a3352 0%, #2d6a9f 100%)',
          borderColor: '#2d6a9f',
        }}
      >
        <div className="absolute top-0 right-0 w-48 h-48 rounded-full opacity-10" style={{ background: '#7ab3d8', transform: 'translate(30%, -30%)' }} />
        <div className="relative z-10">
          <div className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: '#7ab3d8' }}>
            Target Client Influence
          </div>
          <div className="flex items-end gap-6 flex-wrap">
            <div>
              <div className="text-4xl font-display font-bold text-white">35.0%</div>
              <div className="text-sm mt-1" style={{ color: '#b8d8f0' }}>Estimated influence on global model</div>
            </div>
            <div className="flex gap-5 flex-wrap">
              {[
                { label: 'Client', value: 'C-04' },
                { label: 'Contribution', value: '28.0%' },
                { label: 'Model Updates', value: '20' },
                { label: 'Compression', value: 'Low' },
              ].map((m) => (
                <div key={m.label}>
                  <div className="text-xs" style={{ color: 'rgba(184,216,240,0.7)' }}>{m.label}</div>
                  <div className="text-sm font-bold text-white mt-0.5">{m.value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {clients.map((c, i) => (
          <div
            key={c.id}
            className="bg-card rounded-2xl border p-4"
            style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)', borderLeft: `3px solid ${colors[i]}` }}
          >
            <div className="flex items-center gap-2 mb-3">
              <div className="w-6 h-6 rounded-lg flex items-center justify-center" style={{ background: colors[i] + '18', color: colors[i] }}>
                <TrendingUp size={12} />
              </div>
              <span className="font-mono text-xs font-semibold" style={{ color: colors[i] }}>{c.id}</span>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span style={{ color: '#6a8caa' }}>Contribution</span>
                <span className="font-mono font-semibold" style={{ color: '#1a3352' }}>{(c.contributionScore * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between text-xs">
                <span style={{ color: '#6a8caa' }}>Influence</span>
                <span className="font-mono font-semibold" style={{ color: '#1a3352' }}>{(c.influenceScore * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between text-xs">
                <span style={{ color: '#6a8caa' }}>Compression</span>
                <span className="font-mono font-semibold" style={{ color: '#1a3352' }}>{c.compressionLevel}</span>
              </div>
              <div className="h-1 rounded-full mt-2" style={{ background: '#eef3f9' }}>
                <div className="h-1 rounded-full" style={{ width: `${c.influenceScore * 100 / 0.4 * 100}%`, background: colors[i] }} />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div
          className="bg-card rounded-2xl border p-5"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
        >
          <h3 className="font-display font-semibold text-sm mb-4" style={{ color: '#1a3352' }}>Contribution History by Round</h3>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={contributionData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef3f9" />
              <XAxis dataKey="round" tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <Tooltip formatter={(v) => [`${(+(v as number) * 100).toFixed(1)}%`]} contentStyle={{ borderRadius: 8, border: '1px solid #d8e6f2', fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              {['C01', 'C02', 'C03', 'C04', 'C05'].map((k, i) => (
                <Area key={k} type="monotone" dataKey={k} stroke={colors[i]} fill={colors[i] + '22'} strokeWidth={2} name={`C-0${i + 1}`} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div
          className="bg-card rounded-2xl border p-5"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
        >
          <h3 className="font-display font-semibold text-sm mb-4" style={{ color: '#1a3352' }}>Estimated Influence per Client</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={influenceData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef3f9" />
              <XAxis dataKey="client" tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <Tooltip formatter={(v) => [`${(+(v as number) * 100).toFixed(1)}%`]} contentStyle={{ borderRadius: 8, border: '1px solid #d8e6f2', fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Bar dataKey="influence" fill="#2d6a9f" radius={[4, 4, 0, 0]} name="Influence" />
              <Bar dataKey="contribution" fill="#9ecbe8" radius={[4, 4, 0, 0]} name="Contribution" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model update sizes */}
      <div
        className="bg-card rounded-2xl border p-5"
        style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-display font-semibold text-sm" style={{ color: '#1a3352' }}>Model Update Size (MB) per Client per Round</h3>
          <div className="flex gap-3">
            {[
              { icon: Database, label: 'Updates tracked', val: '94' },
              { icon: Layers, label: 'Avg size', val: '1.8 MB' },
              { icon: Maximize2, label: 'Max compression', val: 'High' },
            ].map((m) => (
              <div key={m.label} className="flex items-center gap-1.5 text-xs" style={{ color: '#6a8caa' }}>
                <m.icon size={11} />
                <span>{m.val} {m.label}</span>
              </div>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={updateSizes} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eef3f9" />
            <XAxis dataKey="round" tick={{ fontSize: 10, fill: '#6a8caa' }} />
            <YAxis tick={{ fontSize: 10, fill: '#6a8caa' }} unit=" MB" />
            <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #d8e6f2', fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            {['C01', 'C02', 'C03', 'C04', 'C05'].map((k, i) => (
              <Line key={k} type="monotone" dataKey={k} stroke={colors[i]} strokeWidth={2} dot={false} name={`C-0${i + 1}`} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
