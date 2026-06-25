"""Quick SPC chart for Aircraft Equipment Problem Critical."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import load_and_merge_asrs
from spc import run_spc_pipeline

asrs = pd.read_parquet("outputs/data/asrs_layer1.parquet")
asrs['date'] = pd.to_datetime(asrs['date'], errors='coerce')

CAT = 'Aircraft Equipment Problem Critical'
result = run_spc_pipeline(asrs, category_value=CAT, start_date='2018-01-01')

if result is None:
    print("SPC returned None — check category name or data")
    sys.exit(1)

print(f"Alarms: {len(result['alarms'])}")
print(f"First alarm: {result['alarms'][0] if result['alarms'] else 'none'}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.suptitle(
    'Aircraft Equipment Problem Critical — Monthly Rate + CUSUM\n'
    'Post-COVID return-to-service deferred maintenance spike (May 2022)',
    fontsize=13, fontweight='bold',
)

# Top: monthly counts + STL trend
mc = result['monthly_counts']
ax1.plot(mc.index, mc.values, color='#003366', lw=2,
         alpha=0.6, label='Monthly incidents')
ax1.plot(result['trend'].index, result['trend'].values,
         color='navy', lw=2.5, label='STL trend')
ax1.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-12-01'),
            alpha=0.08, color='grey', label='COVID operations (2020-2021)')
ax1.axvspan(pd.Timestamp('2022-05-01'), pd.Timestamp('2022-12-01'),
            alpha=0.15, color='red', label='SPC alarm — maintenance spike')
for alarm in result['alarms']:
    ax1.axvline(alarm, color='red', lw=1, alpha=0.4)
ax1.set_ylabel('Incidents per month', fontsize=11)
ax1.legend(fontsize=9)
ax1.set_title('Monthly counts and STL trend', fontsize=10)

# Bottom: CUSUM
idx = mc.index
ax2.plot(idx, result['s_pos'], color='#cc0000', lw=2, label='CUSUM S+')
ax2.axhline(result['control_limit'], color='black', ls='--', lw=1.5,
            label=f'Control limit h={result["control_limit"]}')
alarm_above = [s > result['control_limit'] for s in result['s_pos']]
ax2.fill_between(idx, result['s_pos'], result['control_limit'],
                 where=alarm_above, alpha=0.3, color='red', label='ALARM zone')
# Mark first alarm
if result['alarms']:
    first = result['alarms'][0]
    ax2.axvline(first, color='red', lw=2.5, alpha=0.8)
    ax2.text(first, result['control_limit'] + 0.5,
             f"First alarm\n{first.strftime('%b %Y')}",
             fontsize=9, color='red', fontweight='bold')
ax2.set_ylabel('CUSUM statistic', fontsize=11)
ax2.set_xlabel('Date', fontsize=11)
ax2.legend(fontsize=9)
ax2.set_title('CUSUM control chart', fontsize=10)

plt.tight_layout()
out = 'outputs/figures/equipment_critical_spc.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
