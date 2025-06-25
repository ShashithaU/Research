"""
Quick test of the final DP model
"""

import os
from dp_simple_model_tester import load_and_test_model

def quick_test():
    # Paths to your final DP model
    model_path = "saved_models/dp_global_model_dp_final.pth"
    info_path = "saved_models/dp_global_model_dp_final_info.pkl"
    scaler_path = "scalers/global_scaler.pkl"
    test_data_path = "Data/Test/TestData.csv"  # Update this path
    
    print("🔍 Quick DP Model Test")
    print("="*40)
    
    # Check if files exist
    if not os.path.exists(model_path):
        print("❌ DP model not found. Run the DP federated learning first.")
        return
    
    if not os.path.exists(test_data_path):
        print("❌ Test data not found. Update the test_data_path in the script.")
        return
    
    # Test the model
    result = load_and_test_model(model_path, info_path, test_data_path, scaler_path)
    
    if result:
        print(f"\n✅ DP Model Performance:")
        print(f"   Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"   Loss: {result['loss']:.4f}")
        print(f"   Test samples: {len(result['true_labels'])}")
    else:
        print("❌ Testing failed")

if __name__ == "__main__":
    quick_test()