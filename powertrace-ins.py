import sys, os, sim

class PowerTrace:
  def setup(self, args):
    args = dict(enumerate((args or '').split(':')))
    # Default to 500,000 instructions if no argument is provided
    self.interval_ins = int(args.get(0, '') or 500000)
    sim.util.EveryIns(self.interval_ins, self.periodic, roi_only=True)
    self.ins_last = 0

  def periodic(self, ins, ins_delta):
    if ins == 0:
      return
    sim.stats.write(str(ins))  # Write stats file with instruction count prefix
    self.do_power(self.ins_last, ins)
    self.ins_last = ins

  def hook_roi_end(self):
    self.ins_roi_end = int(sim.stats.get("performance_model", 0, "instruction_count"))

  def hook_sim_end(self):
    self.do_power(self.ins_last, None)

  def do_power(self, ins0, ins1):
    _ins0 = 'roi-begin' if ins0 == 0 else 'periodicins-%d' % ins0
    _ins1 = 'roi-end' if ins1 is None else 'periodicins-%d' % ins1
    if ins1 is None:
      ins1 = self.ins_roi_end
    # Run McPAT for this instruction interval
    os.system('unset PYTHONHOME; %s -d %s -o %s --partial=%s:%s --no-graph' % (
      os.path.join(os.getenv('SNIPER_ROOT'), 'tools/mcpat.py'),
      sim.config.output_dir,
      os.path.join(sim.config.output_dir, 'power-%s-%s-%s' % (_ins0, _ins1, ins1 - ins0)),
      _ins0, _ins1
    ))

sim.util.register(PowerTrace())

