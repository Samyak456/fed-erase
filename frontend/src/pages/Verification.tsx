import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts';
import { ShieldCheck, TrendingDown, Activity, GitCompare } from 'lucide-react';
import { verificationData, verificationChartData } from '../data/mockData';

const accuracyComparison = [
  { name: 'Before', fedeRase: 86.3, ideal: 86.3 },
  { name: 'After FedErase', fedeRase: 84.7, ideal: null },
  { name: 'Ideal Retrain', fedeRase: null, ideal: 83.9 },
];

const forgettingComparison = [
  { name: 'FedErase', score: 94.2 },
  { name: 'Ideal Retrain', score: 98.5 },
  { name: 'Checkpoint Rollback', score: 76.4 },
  { name: 'Naive Subtraction', score: 38.1 },
];

const miData = [
  { round: 0, attack: 0.68, random: 0.50 },
  { round: 1, attack: 0.62, random: 0.50 },
  { round: 2, attack: 0.58, random: 0.50 },
  { round: 3, attack: 0.54, random: 0.50 },
  { round: 4, attack: 0.51, random: 0.50 },
  { round: 5, attack: 0.50, random: 0.50 },
  { round: 6, attack: 0.50, random: 0.50 },
];

export default function Verification() {
  const { before, after, ideal } = verificationData;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header banner */}
      <div
        className="rounded-2xl p-5 border flex items-start gap-4"
        style={{ background: 'rgba(45,106,159,0.05)', borderColor: '#d8e6f2' }}
      >
        <div className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0" style={{ background: '#ddeaf7', color: '#2d6a9f' }}>
          <ShieldCheck size={18} />
        </div>
        <div>
          <div className="font-display font-semibold text-sm" style={{ color: '#1a3352' }}>Forgetting Verification Dashboard</div>
          <p className="text-xs mt-1 leading-relaxed" style={{ color: '#6a8caa' }}>
            Verifies that the target client's influence has been removed from the global model while overall model utility is preserved.
            Metrics include forgetting score, client influence residual, model distance, and membership inference test results.
          </p>
        </div>
        <div className="ml-auto shrink-0">
          <span className="px-3 py-1.5 rounded-full text-xs font-semibold" style={{ background: 'rgba(34,197,94,0.12)', color: '#22c55e' }}>
            ✓ Verification Passed
          </span>
        </div>
      </div>

      {/* Before/After comparison cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {[
          { title: 'Before Unlearning', data: before, bg: '#fff7f0', border: '#f0d8c8', label: 'baseline', tag: null },
          { title: 'After FedErase', data: after, bg: '#f0f8ff', border: '#c0daf0', label: 'result', tag: 'FedErase' },
          { title: 'Ideal (Full Retrain)', data: ideal, bg: '#f0fdf4', border: '#b8e6c8', label: 'ideal', tag: 'Reference' },
        ].map(({ title, data, bg, border, tag }) => (
          <div
            key={title}
            className="rounded-2xl border p-5"
            style={{ background: bg, borderColor: border, boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="font-display font-semibold text-sm" style={{ color: '#1a3352' }}>{title}</div>
              {tag && (
                <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{ background: '#ddeaf7', color: '#2d6a9f' }}>
                  {tag}
                </span>
              )}
            </div>
            <div className="space-y-3">
              {[
                { label: 'Model Accuracy', value: `${data.accuracy}%`, icon: Activity },
                { label: 'Forgetting Score', value: `${data.forgettingScore}%`, icon: TrendingDown },
                { label: 'Client Influence', value: `${data.clientInfluence}%`, icon: GitCompare },
                { label: 'Model Distance', value: data.modelDistance.toFixed(3), icon: ShieldCheck },
              ].map((m) => (
                <div key={m.label} className="flex justify-between items-center">
                  <div className="flex items-center gap-1.5 text-xs" style={{ color: '#6a8caa' }}>
                    <m.icon size={11} />
                    {m.label}
                  </div>
                  <span className="text-sm font-bold font-mono" style={{ color: '#1a3352' }}>{m.value}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Charts row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Influence reduction */}
        <div
          className="bg-card rounded-2xl border p-5"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
        >
          <h3 className="font-display font-semibold text-sm mb-4" style={{ color: '#1a3352' }}>Influence Reduction Over Repair Rounds</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={verificationChartData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef3f9" />
              <XAxis dataKey="round" tick={{ fontSize: 10, fill: '#6a8caa' }} label={{ value: 'Repair Round', position: 'insideBottom', offset: -2, fontSize: 10, fill: '#6a8caa' }} />
              <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <Tooltip formatter={(v) => [`${(+(v as number) * 100).toFixed(1)}%`]} contentStyle={{ borderRadius: 8, border: '1px solid #d8e6f2', fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Line type="monotone" dataKey="fedeRase" stroke="#2d6a9f" strokeWidth={2.5} dot={{ fill: '#2d6a9f', r: 3 }} name="FedErase" />
              <Line type="monotone" dataKey="ideal" stroke="#22c55e" strokeWidth={1.5} strokeDasharray="4 4" dot={false} name="Ideal" />
              <Line type="monotone" dataKey="naive" stroke="#e05c5c" strokeWidth={1.5} strokeDasharray="4 4" dot={false} name="Naive Sub." />
              <ReferenceLine y={0.05} stroke="#b8d8f0" strokeDasharray="3 3" label={{ value: 'Target threshold', fontSize: 9, fill: '#a0bdd0' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Forgetting comparison */}
        <div
          className="bg-card rounded-2xl border p-5"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
        >
          <h3 className="font-display font-semibold text-sm mb-4" style={{ color: '#1a3352' }}>Forgetting Score Comparison</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={forgettingComparison} layout="vertical" margin={{ top: 4, right: 12, bottom: 0, left: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef3f9" horizontal={false} />
              <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: '#6a8caa' }} width={80} />
              <Tooltip formatter={(v) => [`${v}%`]} contentStyle={{ borderRadius: 8, border: '1px solid #d8e6f2', fontSize: 12 }} />
              <Bar dataKey="score" radius={[0, 4, 4, 0]}
                fill="#2d6a9f"
                label={{ position: 'right', fontSize: 10, fill: '#6a8caa', formatter: (v: unknown) => `${v}%` }}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Membership inference + accuracy */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div
          className="bg-card rounded-2xl border p-5"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
        >
          <h3 className="font-display font-semibold text-sm mb-1" style={{ color: '#1a3352' }}>Membership Inference Attack</h3>
          <p className="text-xs mb-4" style={{ color: '#6a8caa' }}>Attack accuracy converges to random (0.5) after unlearning — confirming forgetting.</p>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={miData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef3f9" />
              <XAxis dataKey="round" tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <YAxis domain={[0.45, 0.75]} tickFormatter={(v) => v.toFixed(2)} tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #d8e6f2', fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Line type="monotone" dataKey="attack" stroke="#e05c5c" strokeWidth={2} dot={{ fill: '#e05c5c', r: 3 }} name="MI Attack Acc." />
              <Line type="monotone" dataKey="random" stroke="#a0bdd0" strokeWidth={1} strokeDasharray="4 4" dot={false} name="Random Baseline" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div
          className="bg-card rounded-2xl border p-5"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
        >
          <h3 className="font-display font-semibold text-sm mb-1" style={{ color: '#1a3352' }}>Accuracy Comparison</h3>
          <p className="text-xs mb-4" style={{ color: '#6a8caa' }}>Model utility preserved after unlearning — accuracy stays close to the ideal retrained model.</p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart
              data={[
                { name: 'Before', val: 86.3 },
                { name: 'After FedErase', val: 84.7 },
                { name: 'Ideal Retrain', val: 83.9 },
                { name: 'Checkpoint\nRollback', val: 81.0 },
              ]}
              margin={{ top: 4, right: 8, bottom: 0, left: -20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#eef3f9" />
              <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#6a8caa' }} />
              <YAxis domain={[75, 90]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <Tooltip formatter={(v) => [`${v}%`]} contentStyle={{ borderRadius: 8, border: '1px solid #d8e6f2', fontSize: 12 }} />
              <Bar dataKey="val" fill="#2d6a9f" radius={[4, 4, 0, 0]} name="Accuracy" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
