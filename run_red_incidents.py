"""Pull top 20 RED quadrant incidents by IF score and print top 5 narratives."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import pandas as pd

asrs = pd.read_parquet("outputs/data/asrs_layer1.parquet")
asrs['date'] = pd.to_datetime(asrs['date'], errors='coerce')

red_top = (asrs[asrs['quadrant'] == 'RED']
           .nlargest(20, 'if_score')
           [['ACN', 'date', 'Events | Anomaly', 'Aircraft 1 | Make Model Name',
             'Aircraft 1 | Flight Phase', 'Assessments | Primary Problem',
             'if_score', 'spc_flag', 'full_narrative']]
           .copy())

print(f"Total RED incidents: {(asrs['quadrant'] == 'RED').sum():,}")
print(f"\nTop 20 by IF score:")
print(red_top[['ACN', 'date', 'if_score', 'Events | Anomaly',
               'Aircraft 1 | Make Model Name', 'Aircraft 1 | Flight Phase']].to_string())

print("\n" + "=" * 70)
print("TOP 5 NARRATIVES (highest novelty + SPC alarm):")
print("=" * 70)
for i, (_, row) in enumerate(red_top.head(5).iterrows(), 1):
    date_str = row['date'].date() if pd.notna(row['date']) else 'unknown'
    print(f"\n[{i}] ACN: {row['ACN']}  |  {date_str}  |  IF score: {row['if_score']:.4f}")
    print(f"    Aircraft: {row['Aircraft 1 | Make Model Name']}")
    print(f"    Phase:    {row['Aircraft 1 | Flight Phase']}")
    print(f"    Anomaly:  {str(row['Events | Anomaly'])[:120]}")
    print(f"    Problem:  {row['Assessments | Primary Problem']}")
    print(f"    Narrative:")
    narrative = str(row['full_narrative'])
    # Print in 100-char lines for readability
    for j in range(0, min(600, len(narrative)), 100):
        print(f"      {narrative[j:j+100]}")
    print("-" * 70)

# Save to CSV for reference
red_top.drop(columns=['full_narrative']).to_csv(
    'outputs/data/red_top20_incidents.csv', index=False)
print("\nSaved: outputs/data/red_top20_incidents.csv")
