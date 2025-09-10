# TOP-3 CONFIGURATION PREDICTOR WITH PPW COST ANALYSIS
# Add this to a new cell

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
csv_filename = 'trainTop3.csv'  # Replace with your actual file path

print("🚀 Random Forest Configuration Prediction Model")
print("=" * 50)

# Load the dataset
try:
    df = pd.read_csv(csv_filename)
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")
    print("\n🔍 First few rows:")
    print(df.head())
except FileNotFoundError:
    print("❌ CSV file not found. Please check the file path.")

print("🎯 TOP-3 CONFIGURATION PREDICTOR")
print("=" * 60)

# Configuration
TARGET_COLUMNS = ['btbCore0_best', 'btbCore1_best']  # What we're trying to predict
ALL_CONFIG_COLUMNS = [
    'btbCore0_best', 'btbCore1_best', 'PPW_best',
    'btbCore0_2nd', 'btbCore1_2nd', 'PPW_2nd', 'Diff_best_2nd',
    'btbCore0_3rd', 'btbCore1_3rd', 'PPW_3rd', 'Diff_best_3rd'
]

# Check if we have the required columns
missing_cols = [col for col in ALL_CONFIG_COLUMNS if col not in df.columns]
if missing_cols:
    print(f"❌ Missing columns: {missing_cols}")
    print(f"Available columns: {[col for col in df.columns if 'btb' in col.lower() or 'ppw' in col.lower()]}")
else:
    print("✅ All required columns found!")

    # 1. DATA PREPARATION
    print(f"\n📊 DATA PREPARATION")
    print("-" * 40)

    # Exclude target and performance columns from features
    X = df.drop(ALL_CONFIG_COLUMNS, axis=1)
    y = df[TARGET_COLUMNS].copy()

    # Store top-3 configurations and performance data
    top3_configs = {
        'best': df[['btbCore0_best', 'btbCore1_best', 'PPW_best']].copy(),
        '2nd': df[['btbCore0_2nd', 'btbCore1_2nd', 'PPW_2nd']].copy(),
        '3rd': df[['btbCore0_3rd', 'btbCore1_3rd', 'PPW_3rd']].copy()
    }

    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"Target 1 unique values: {len(y.iloc[:, 0].unique())}")
    print(f"Target 2 unique values: {len(y.iloc[:, 1].unique())}")

    # Show performance statistics
    print(f"\n📈 PERFORMANCE STATISTICS")
    print("-" * 40)
    for rank in ['best', '2nd', '3rd']:
        ppw_col = f'PPW_{rank}'
        if ppw_col in df.columns:
            # Convert to integer first, then to float for calculations
            #df[ppw_col] = df[ppw_col].astype(str).astype('int64').astype('float64')

            ppw_mean = df[ppw_col].mean()
            ppw_std = df[ppw_col].std()
            print(f"{rank.upper():>4} PPW: {ppw_mean:.4f} ± {ppw_std:.4f}")

    # Show configuration frequency
    print(f"\n📊 CONFIGURATION FREQUENCY")
    print("-" * 40)
    for rank in ['best', '2nd', '3rd']:
        config_combo = df[f'btbCore0_{rank}'].astype(str) + '-' + df[f'btbCore1_{rank}'].astype(str)
        top_configs = config_combo.value_counts().head(5)
        print(f"\nTop 5 {rank.upper()} configurations:")
        for config, count in top_configs.items():
            print(f"  {config}: {count} ({count/len(df)*100:.1f}%)")

    # 2. FEATURE PREPROCESSING
    print(f"\n🔧 FEATURE PREPROCESSING")
    print("-" * 40)

    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded: {col}")

    # Remove non-numeric columns that couldn't be encoded
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    print(f"Final feature set: {X.shape}")

    # 3. TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Get corresponding top-3 data for test set
    test_indices = X_test.index
    test_top3 = {}
    for rank, data in top3_configs.items():
        test_top3[rank] = data.loc[test_indices]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. MODEL TRAINING
    print(f"\n🌲 MODEL TRAINING")
    print("-" * 40)

    models = {}
    predictions = {}

    for i, target_col in enumerate(TARGET_COLUMNS):
        print(f"Training {target_col}...")

        # Get the target column
        y_target = y_train.iloc[:, i]

        # Find rows where target is not missing
        valid_mask = y_target.notna()

        if valid_mask.sum() == 0:
            print(f"No valid data for {target_col}, skipping...")
            continue

        print(f"Using {valid_mask.sum()} samples out of {len(y_target)} total")

        # Use only valid rows for both features and target
        X_train_valid = X_train_scaled[valid_mask]
        y_train_valid = y_target[valid_mask]

        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        )

        rf.fit(X_train_valid, y_train_valid)

        models[target_col] = rf
        predictions[target_col] = rf.predict(X_test_scaled)

        # Individual accuracy
        accuracy = accuracy_score(y_test.iloc[:, i], predictions[target_col])
        print(f"  {target_col} accuracy: {accuracy:.4f}")

    # 5. TOP-K ACCURACY EVALUATION
    print(f"\n🏆 TOP-K ACCURACY EVALUATION")
    print("-" * 40)

    exact_matches = 0
    top3_matches = 0
    ppw_costs = {'exact': [], 'top3_miss': [], 'top3_hit': []}
    detailed_results = []

    for i in range(len(X_test)):
        # Predicted configuration
        pred_core0 = predictions[TARGET_COLUMNS[0]][i]
        pred_core1 = predictions[TARGET_COLUMNS[1]][i]
        pred_config = (pred_core0, pred_core1)

        # Actual configurations (best, 2nd, 3rd)
        actual_configs = []
        for rank in ['best', '2nd', '3rd']:
            actual_core0 = test_top3[rank].iloc[i][f'btbCore0_{rank}']
            actual_core1 = test_top3[rank].iloc[i][f'btbCore1_{rank}']
            actual_ppw = test_top3[rank].iloc[i][f'PPW_{rank}']
            actual_configs.append({
                'rank': rank,
                'config': (actual_core0, actual_core1),
                'ppw': actual_ppw
            })

        # Check matches
        exact_match = pred_config == actual_configs[0]['config']
        top3_match = any(pred_config == cfg['config'] for cfg in actual_configs)

        if exact_match:
            exact_matches += 1
            ppw_costs['exact'].append(actual_configs[0]['ppw'])


        if top3_match:
            top3_matches += 1
            # Find which rank was matched
            matched_rank = next(cfg['rank'] for cfg in actual_configs if pred_config == cfg['config'])
            matched_ppw = next(cfg['ppw'] for cfg in actual_configs if pred_config == cfg['config'])
            ppw_costs['top3_hit'].append(matched_ppw)
        else:
            ppw_costs['top3_miss'].append(actual_configs[0]['ppw'])  # Cost of missing = best PPW

        # Store detailed result
        detailed_results.append({
            'sample': i,
            'predicted': pred_config,
            'actual_best': actual_configs[0]['config'],
            'actual_2nd': actual_configs[1]['config'],
            'actual_3rd': actual_configs[2]['config'],
            'ppw_best': actual_configs[0]['ppw'],
            'ppw_2nd': actual_configs[1]['ppw'],
            'ppw_3rd': actual_configs[2]['ppw'],
            'exact_match': exact_match,
            'top3_match': top3_match
        })

    # Calculate accuracies
    exact_accuracy = exact_matches / len(X_test)
    top3_accuracy = top3_matches / len(X_test)

    print(f"📊 ACCURACY RESULTS:")
    print(f"  Exact Match (Best Config):  {exact_accuracy:.4f} ({exact_matches}/{len(X_test)})")
    print(f"  Top-3 Match (Any of 3):     {top3_accuracy:.4f} ({top3_matches}/{len(X_test)})")
    print(f"  Improvement:                +{(top3_accuracy - exact_accuracy):.4f} ({top3_matches - exact_matches} more correct)")

    # 6. PPW COST ANALYSIS
    print(f"\n💰 PPW COST ANALYSIS")
    print("-" * 40)

    # Calculate average PPW costs
    if ppw_costs['exact']:
        avg_ppw_exact = np.mean(ppw_costs['exact'])
        print(f"Average PPW when exact match:     {avg_ppw_exact:.6f}")

    if ppw_costs['top3_hit']:
        avg_ppw_top3 = np.mean(ppw_costs['top3_hit'])
        print(f"Average PPW when top-3 match:     {avg_ppw_top3:.6f}")

    if ppw_costs['top3_miss']:
        avg_ppw_miss = np.mean(ppw_costs['top3_miss'])
        print(f"Average PPW when missed (best):   {avg_ppw_miss:.6f}")

    # Calculate performance cost of relaxed matching
    if ppw_costs['exact'] and ppw_costs['top3_hit']:
        performance_cost = np.mean(ppw_costs['top3_hit']) - np.mean(ppw_costs['exact'])
        print(f"Performance cost of top-3 match:  {performance_cost:.6f} PPW ({performance_cost/np.mean(ppw_costs['exact'])*100:.2f}%)")

    # 7. VISUALIZATIONS
    print(f"\n📈 GENERATING VISUALIZATIONS")
    print("-" * 40)

    # Accuracy comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Accuracy comparison
    accuracies = [exact_accuracy, top3_accuracy]
    labels = ['Exact Match\n(Best Only)', 'Top-3 Match\n(Any of 3)']
    colors = ['#ff7f0e', '#2ca02c']

    bars = axes[0].bar(labels, accuracies, color=colors, alpha=0.7)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylim(0, 1)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: PPW distribution
    ppw_data = []
    ppw_labels = []

    if ppw_costs['exact']:
        ppw_data.append(ppw_costs['exact'])
        ppw_labels.append(f'Exact Match\n(n={len(ppw_costs["exact"])})')

    if ppw_costs['top3_hit']:
        ppw_data.append(ppw_costs['top3_hit'])
        ppw_labels.append(f'Top-3 Match\n(n={len(ppw_costs["top3_hit"])})')

    if ppw_costs['top3_miss']:
        ppw_data.append(ppw_costs['top3_miss'])
        ppw_labels.append(f'Missed\n(n={len(ppw_costs["top3_miss"])})')

    axes[1].boxplot(ppw_data, labels=ppw_labels)
    axes[1].set_ylabel('PPW (Performance Per Watt)')
    axes[1].set_title('PPW Distribution by Match Type')
    axes[1].tick_params(axis='x', rotation=45)

    # Plot 3: Sample predictions
    sample_size = min(20, len(detailed_results))
    sample_results = detailed_results[:sample_size]

    exact_matches_sample = [r['exact_match'] for r in sample_results]
    top3_matches_sample = [r['top3_match'] for r in sample_results]

    x = range(sample_size)
    width = 0.35

    axes[2].bar([i - width/2 for i in x], exact_matches_sample, width,
               label='Exact Match', alpha=0.7, color='#ff7f0e')
    axes[2].bar([i + width/2 for i in x], top3_matches_sample, width,
               label='Top-3 Match', alpha=0.7, color='#2ca02c')

    axes[2].set_xlabel('Sample Index')
    axes[2].set_ylabel('Match (1=Yes, 0=No)')
    axes[2].set_title(f'First {sample_size} Predictions')
    axes[2].legend()
    axes[2].set_xticks(x)

    plt.tight_layout()
    plt.show()

    # 8. DETAILED RESULTS
    print(f"\n📋 SAMPLE DETAILED RESULTS")
    print("-" * 60)

    for i in range(min(10, len(detailed_results))):
        result = detailed_results[i]
        print(f"\nSample {i+1}:")
        print(f"  Predicted:    {result['predicted']}")
        print(f"  Actual Best:  {result['actual_best']} (PPW: {result['ppw_best']:.6f})")
        print(f"  Actual 2nd:   {result['actual_2nd']} (PPW: {result['ppw_2nd']:.6f})")
        print(f"  Actual 3rd:   {result['actual_3rd']} (PPW: {result['ppw_3rd']:.6f})")
        print(f"  Exact Match:  {'✅' if result['exact_match'] else '❌'}")
        print(f"  Top-3 Match:  {'✅' if result['top3_match'] else '❌'}")

    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"Key Insights:")
    print(f"• Relaxing from exact to top-3 matching improves accuracy by {(top3_accuracy-exact_accuracy)*100:.1f} percentage points")
    print(f"• {top3_matches - exact_matches} additional correct predictions when considering top-3")
    if ppw_costs['exact'] and ppw_costs['top3_hit']:
        cost_pct = (np.mean(ppw_costs['top3_hit']) - np.mean(ppw_costs['exact'])) / np.mean(ppw_costs['exact']) * 100
        print(f"• Performance cost of top-3 matching: {cost_pct:.2f}% PPW reduction")
