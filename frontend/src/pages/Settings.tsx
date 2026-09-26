import { useState } from 'react';
import { Save, Settings as SettingsIcon, Bell, Server, Database, Shield } from 'lucide-react';

function Section({ title, icon: Icon, children }: { title: string; icon: React.ComponentType<{ size?: number }>; children: React.ReactNode }) {
  return (
    <div
      className="bg-card rounded-2xl border p-6"
      style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
    >
      <div className="flex items-center gap-2.5 mb-5">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: '#ddeaf7', color: '#2d6a9f' }}>
          <Icon size={15} />
        </div>
        <h3 className="font-display font-semibold text-sm" style={{ color: '#1a3352' }}>{title}</h3>
      </div>
      {children}
    </div>
  );
}

function Row({ label, sub, children }: { label: string; sub?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 py-3 border-b last:border-0" style={{ borderColor: '#eef3f9' }}>
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium" style={{ color: '#1a3352' }}>{label}</div>
        {sub && <div className="text-xs mt-0.5" style={{ color: '#6a8caa' }}>{sub}</div>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

function Toggle({ defaultOn = false }: { defaultOn?: boolean }) {
  const [on, setOn] = useState(defaultOn);
  return (
    <button
      onClick={() => setOn((v) => !v)}
      className="relative w-10 h-5 rounded-full cursor-pointer transition-colors"
      style={{ background: on ? '#2d6a9f' : '#d8e6f2' }}
    >
      <span
        className="absolute top-0.5 w-4 h-4 rounded-full bg-white shadow-sm transition-transform"
        style={{ transform: on ? 'translateX(22px)' : 'translateX(2px)' }}
      />
    </button>
  );
}

function Input({ defaultValue, mono = false }: { defaultValue: string; mono?: boolean }) {
  return (
    <input
      defaultValue={defaultValue}
      className="border rounded-lg px-3 py-1.5 text-sm w-44"
      style={{
        borderColor: 'var(--color-border)',
        color: '#1a3352',
        fontFamily: mono ? 'var(--font-mono)' : 'inherit',
        outline: 'none',
        background: '#fff',
      }}
    />
  );
}

function Select({ options, defaultValue }: { options: string[]; defaultValue: string }) {
  return (
    <select
      defaultValue={defaultValue}
      className="border rounded-lg px-3 py-1.5 text-sm cursor-pointer appearance-none"
      style={{ borderColor: 'var(--color-border)', color: '#1a3352', background: '#fff', outline: 'none', minWidth: 140 }}
    >
      {options.map((o) => <option key={o}>{o}</option>)}
    </select>
  );
}

export default function Settings() {
  const [saved, setSaved] = useState(false);

  function handleSave() {
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  }

  return (
    <div className="space-y-5 animate-fade-in max-w-3xl">
      <Section title="Model Configuration" icon={Server}>
        <Row label="Aggregation Strategy" sub="How client model updates are combined each round.">
          <Select options={['FedAvg', 'FedProx', 'FedMedian', 'Krum']} defaultValue="FedAvg" />
        </Row>
        <Row label="Training Rounds" sub="Total number of federated learning rounds.">
          <Input defaultValue="20" mono />
        </Row>
        <Row label="Local Epochs" sub="Number of local training epochs per client per round.">
          <Input defaultValue="5" mono />
        </Row>
        <Row label="Learning Rate" sub="Global optimizer learning rate.">
          <Input defaultValue="0.001" mono />
        </Row>
        <Row label="Minimum Clients" sub="Minimum clients required to participate per round.">
          <Input defaultValue="3" mono />
        </Row>
      </Section>

      <Section title="Unlearning Configuration" icon={SettingsIcon}>
        <Row label="Influence Estimation Method" sub="Gradient-based method for estimating client contribution.">
          <Select options={['Gradient Approximation', 'Shapley Estimation', 'IF (TracIn)', 'LISSA']} defaultValue="Gradient Approximation" />
        </Row>
        <Row label="Default Unlearning Strategy" sub="Fallback strategy when adaptive policy is disabled.">
          <Select options={['Deep Unlearning', 'Standard Unlearning', 'Lightweight Unlearning']} defaultValue="Deep Unlearning" />
        </Row>
        <Row label="Forgetting Threshold" sub="Minimum forgetting score to pass verification.">
          <Input defaultValue="0.90" mono />
        </Row>
        <Row label="Max Repair Rounds" sub="Upper limit on post-unlearning repair training rounds.">
          <Input defaultValue="5" mono />
        </Row>
        <Row label="Enable Auto-Verify" sub="Automatically run forgetting verification after unlearning.">
          <Toggle defaultOn />
        </Row>
      </Section>

      <Section title="Resource Adaptation Policy" icon={Database}>
        <Row label="Enable Adaptive Policy" sub="Select unlearning strategy based on client resources.">
          <Toggle defaultOn />
        </Row>
        <Row label="High Resource Threshold (CPU)" sub="Minimum CPU % to qualify for Deep Unlearning.">
          <Input defaultValue="80" mono />
        </Row>
        <Row label="Medium Resource Threshold (CPU)" sub="Minimum CPU % to qualify for Standard Unlearning.">
          <Input defaultValue="50" mono />
        </Row>
        <Row label="Default Compression Level" sub="Compression when adaptive policy is off.">
          <Select options={['Low', 'Medium', 'High']} defaultValue="Medium" />
        </Row>
      </Section>

      <Section title="Checkpoint Settings" icon={Database}>
        <Row label="Checkpoint Frequency (Deep)" sub="Steps between checkpoints for Deep Unlearning.">
          <Input defaultValue="1" mono />
        </Row>
        <Row label="Checkpoint Frequency (Standard)" sub="Steps between checkpoints for Standard Unlearning.">
          <Input defaultValue="2" mono />
        </Row>
        <Row label="Checkpoint Frequency (Lightweight)" sub="Steps between checkpoints for Lightweight Unlearning.">
          <Input defaultValue="3" mono />
        </Row>
        <Row label="Max Stored Checkpoints" sub="Maximum number of checkpoints to retain on disk.">
          <Input defaultValue="10" mono />
        </Row>
        <Row label="Auto-Rollback on Failure" sub="Restore last checkpoint if unlearning fails verification.">
          <Toggle defaultOn />
        </Row>
      </Section>

      <Section title="Notifications" icon={Bell}>
        <Row label="Unlearning Complete" sub="Notify when the unlearning pipeline finishes.">
          <Toggle defaultOn />
        </Row>
        <Row label="Client Disconnection" sub="Alert when a client leaves the network.">
          <Toggle defaultOn />
        </Row>
        <Row label="Verification Failed" sub="Alert when forgetting verification does not pass.">
          <Toggle defaultOn />
        </Row>
        <Row label="Low Resource Warning" sub="Warn when a client drops below the low-resource threshold.">
          <Toggle />
        </Row>
        <Row label="Round Completion" sub="Notify after each training round.">
          <Toggle />
        </Row>
      </Section>

      <Section title="Security & Privacy" icon={Shield}>
        <Row label="Differential Privacy" sub="Enable DP noise injection during training.">
          <Toggle />
        </Row>
        <Row label="Secure Aggregation" sub="Use cryptographic secure aggregation protocol.">
          <Toggle defaultOn />
        </Row>
        <Row label="Audit Log Retention" sub="Number of days to retain unlearning audit logs.">
          <Input defaultValue="90" mono />
        </Row>
      </Section>

      {/* Save */}
      <div className="flex justify-end pb-4">
        <button
          onClick={handleSave}
          className="px-6 py-2.5 rounded-xl text-sm font-semibold text-white cursor-pointer flex items-center gap-2"
          style={{ background: saved ? '#22c55e' : '#2d6a9f' }}
        >
          <Save size={15} />
          {saved ? 'Saved!' : 'Save Changes'}
        </button>
      </div>
    </div>
  );
}
