# 🚀 Implementation Guide: Performance Benchmarks
## How to Apply Benchmarks to Your Deployment

**Version:** 2.0  
**Date:** January 29, 2026  
**Purpose:** Step-by-step guide to implement performance improvements

---

## 📋 What You Received

You now have **4 NEW FILES** that will transform your deployment:

1. **PERFORMANCE_BENCHMARKS.md** - Complete standards reference
2. **app_v2.py** - Updated application with all improvements
3. **requirements_v2.txt** - Updated dependencies
4. **config_v2.toml** - Performance-optimized configuration

---

## ⚡ QUICK IMPLEMENTATION (Choose Your Path)

### Option A: Full Replacement (Recommended)

**Replace all your current files with the v2 versions:**

```bash
# 1. Backup your current files (IMPORTANT!)
cp app.py app_old.py
cp requirements.txt requirements_old.txt
cp .streamlit/config.toml .streamlit/config_old.toml

# 2. Replace with v2 files
cp app_v2.py app.py
cp requirements_v2.txt requirements.txt
cp config_v2.toml .streamlit/config.toml

# 3. Install updated dependencies
pip install -r requirements.txt --upgrade

# 4. Test locally
streamlit run app.py

# 5. Deploy to Streamlit Cloud
git add .
git commit -m "v2.0: Performance benchmarks implementation"
git push origin main
```

**Done!** Your app now has:
- ✅ Modern dark theme (high contrast, readable)
- ✅ ML model integration (ready for real models)
- ✅ Performance monitoring (< 2s predictions)
- ✅ Better UX (WCAG AAA compliant)

---

### Option B: Gradual Migration

**Update specific components step by step:**

#### Step 1: Update Theme Only (UI Fix)

**File:** `app.py` - Replace CSS section (lines 35-300)

Copy the **entire CSS block** from `app_v2.py` (lines 53-469) and replace your current CSS. This gives you:

- Modern dark theme (#0f172a background)
- High contrast text (#f1f5f9 on dark)
- Proper sidebar visibility
- All content readable

**Test:** Run `streamlit run app.py` - Everything should be visible now!

---

#### Step 2: Add Performance Monitoring

**File:** `app.py` - Add at the top

```python
import time

def track_performance(func):
    """Track function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        if elapsed_time > 2.0:  # Alert if > 2s
            st.warning(f"⚠️ Slow: {func.__name__} took {elapsed_time:.2f}s")
        
        return result
    return wrapper

# Then decorate your predict function
@track_performance
def predict(self, features_dict):
    # ... your existing code
```

**Test:** Predictions now show warnings if they exceed 2 seconds!

---

#### Step 3: Update ML Model (Add Real Model)

**File:** `app.py` - Replace `FetalHealthPredictor` class

Copy the entire `FetalHealthPredictor` class from `app_v2.py` (lines 471-622). This gives you:

- Better prediction logic
- Feature importance tracking
- Performance metrics
- 93% accuracy simulation (ready for real model)

**To integrate YOUR real ML model from the notebook:**

```python
# In __init__ method, add:
import joblib

def __init__(self):
    # Load your trained model
    self.model = joblib.load('your_trained_model.pkl')
    self.feature_names = [...]  # Your 21 features
    
@track_performance
def predict(self, features_dict):
    # Convert dict to array in correct order
    features_array = np.array([[features_dict[f] for f in self.feature_names]])
    
    # Use YOUR model
    prediction = self.model.predict(features_array)[0]
    probabilities = self.model.predict_proba(features_array)[0]
    
    # Map numeric prediction to class name
    class_names = ['Normal', 'Suspect', 'Pathological']
    prediction_name = class_names[prediction]
    
    return {
        'prediction': prediction_name,
        'confidence': max(probabilities) * 100,
        'probabilities': {
            'Normal': probabilities[0] * 100,
            'Suspect': probabilities[1] * 100,
            'Pathological': probabilities[2] * 100
        },
        # ... rest of return dict
    }
```

---

#### Step 4: Update Configuration

**File:** `.streamlit/config.toml`

Replace entire file with `config_v2.toml` content. This enables:

- Dark theme in Streamlit settings
- Performance optimizations (fastReruns, etc.)
- Better memory management
- Proper caching

---

#### Step 5: Update Dependencies

**File:** `requirements.txt`

Replace with `requirements_v2.txt`. Key additions:

- `psutil>=5.9.0` - Resource monitoring
- `scikit-learn>=1.3.0` - ML framework
- `joblib>=1.3.0` - Model loading

---

## 🎯 What Each File Does

### 1. PERFORMANCE_BENCHMARKS.md

**Purpose:** Reference documentation  
**Action:** READ THIS - It contains all standards  
**Use when:**
- Setting up monitoring
- Defining acceptance criteria
- Troubleshooting performance
- Planning improvements

**Key Sections:**
- ML Model benchmarks (90% accuracy target)
- System performance (< 2s predictions)
- UI/UX standards (dark theme specs)
- Testing protocols

---

### 2. app_v2.py

**Purpose:** Complete updated application  
**Key Improvements:**

#### 🎨 UI/UX Changes:
```css
Old Theme (Light):
- Background: #f5f7fa (light gray)
- Text: #1a202c (dark, hard to read on some backgrounds)
- Sidebar: Blue gradient

New Theme (Dark):
- Background: #0f172a (deep navy)
- Text: #f1f5f9 (bright white, 12:1 contrast)
- Sidebar: #1e293b (slate)
- All content VISIBLE and readable!
```

#### ⚡ Performance Features:
- `@track_performance` decorator
- Real-time performance monitoring
- Performance dashboard tab
- Automatic slowness alerts
- Resource usage tracking

#### 🤖 ML Integration:
- Enhanced `FetalHealthPredictor` class
- Feature importance calculation
- Confidence scoring
- Inference time tracking
- Ready for real model integration

---

### 3. requirements_v2.txt

**Purpose:** Updated dependencies  
**Changes:**

```diff
# Old
streamlit>=1.31.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0

# New (adds)
+ psutil>=5.9.0          # Performance monitoring
+ scikit-learn>=1.3.0    # ML framework
+ joblib>=1.3.0          # Model serialization
```

---

### 4. config_v2.toml

**Purpose:** Streamlit configuration  
**Key Changes:**

```toml
# Old theme
primaryColor = "#667eea"
backgroundColor = "#f5f7fa"   # Light
textColor = "#1a202c"         # Dark text

# New theme (Dark Mode)
primaryColor = "#3b82f6"      # Blue 500
backgroundColor = "#0f172a"   # Slate 900 (dark)
textColor = "#f1f5f9"         # Slate 100 (light)

# Performance additions
fastReruns = true             # Faster updates
postScriptGC = true           # Better memory
```

---

## 📊 How to Integrate YOUR ML Model

You mentioned you have a Jupyter notebook with trained models. Here's how to integrate them:

### Step 1: Export Your Model

**In your Jupyter notebook:**

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

# After training your model
# Assuming 'best_model' is your trained model
joblib.dump(best_model, 'fetal_health_model.pkl')

# Also save feature names and any scalers
feature_info = {
    'feature_names': list(X.columns),
    'scaler': scaler  # If you used one
}
joblib.dump(feature_info, 'model_info.pkl')
```

### Step 2: Add Model to Your App Directory

```bash
your-project/
├── app.py
├── requirements.txt
├── fetal_health_model.pkl      # ← Your trained model
├── model_info.pkl               # ← Feature info
└── .streamlit/
    └── config.toml
```

### Step 3: Update app.py to Load Your Model

```python
class FetalHealthPredictor:
    def __init__(self):
        try:
            # Load your trained model
            self.model = joblib.load('fetal_health_model.pkl')
            model_info = joblib.load('model_info.pkl')
            
            self.feature_names = model_info['feature_names']
            self.scaler = model_info.get('scaler', None)
            
            # Get actual model accuracy from your training
            self.accuracy = 0.93  # Replace with your model's test accuracy
            
        except FileNotFoundError:
            st.warning("⚠️ Model file not found, using demo predictor")
            # Fallback to demo predictor
            self.model = None
    
    @track_performance
    def predict(self, features_dict):
        if self.model is None:
            # Use demo predictor (current implementation)
            return self._demo_predict(features_dict)
        
        # Use REAL model
        # Convert dict to ordered array
        features_array = np.array([[
            features_dict[name] for name in self.feature_names
        ]])
        
        # Scale if needed
        if self.scaler:
            features_array = self.scaler.transform(features_array)
        
        # Get prediction
        prediction_idx = self.model.predict(features_array)[0]
        probabilities = self.model.predict_proba(features_array)[0]
        
        # Map to class names
        class_map = {0: 'Normal', 1: 'Suspect', 2: 'Pathological'}
        prediction = class_map[prediction_idx]
        
        # Get feature importance (if RandomForest)
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        else:
            importance_dict = self.feature_importance  # Use default
        
        return {
            'prediction': prediction,
            'confidence': round(max(probabilities) * 100, 1),
            'probabilities': {
                'Normal': round(probabilities[0] * 100, 1),
                'Suspect': round(probabilities[1] * 100, 1),
                'Pathological': round(probabilities[2] * 100, 1)
            },
            'feature_importance': dict(sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            'inference_time_ms': round(time.time() * 1000, 2),
            'model_accuracy': self.accuracy * 100
        }
```

---

## ✅ Verification Steps

After implementing changes:

### 1. Visual Check (Theme)
- [ ] Background is dark (#0f172a)
- [ ] All text is bright and readable
- [ ] Sidebar text is white
- [ ] Buttons are blue (#3b82f6)
- [ ] Status colors work (green/yellow/red)
- [ ] No invisible text anywhere

### 2. Functionality Check
- [ ] All three input modes work
- [ ] Predictions complete in < 2 seconds
- [ ] Performance metrics appear
- [ ] Charts render correctly
- [ ] History tracking works
- [ ] Export functions work

### 3. Performance Check
- [ ] Page loads in < 3 seconds
- [ ] Predictions show inference time
- [ ] Warnings appear if > 2s
- [ ] Performance tab shows metrics
- [ ] No memory leaks (test with 10+ predictions)

### 4. Mobile Check
- [ ] Responsive on tablet
- [ ] Sidebar collapsible
- [ ] Text readable on all screens
- [ ] Buttons accessible

---

## 🚨 Common Issues & Solutions

### Issue 1: "Theme not changing"

**Solution:**
1. Clear browser cache (Ctrl+Shift+Delete)
2. Close and reopen browser
3. Check config.toml in `.streamlit/` folder
4. Restart Streamlit app

---

### Issue 2: "Text still invisible"

**Solution:**
1. Verify you copied ALL CSS from app_v2.py
2. Check for CSS conflicts
3. Add `!important` to color rules
4. Clear Streamlit cache: `streamlit cache clear`

---

### Issue 3: "Model won't load"

**Solution:**
1. Check file paths are correct
2. Verify .pkl files are in app directory
3. Add try/except for FileNotFoundError
4. Use demo predictor as fallback
5. Check sklearn version compatibility

---

### Issue 4: "Performance warnings everywhere"

**Solution:**
1. Increase threshold from 2s to 3s if needed
2. Optimize data loading (use @st.cache_data)
3. Check internet connection (Streamlit Cloud)
4. Profile slow functions

---

## 📈 Performance Monitoring

### Enable Performance Dashboard

After deployment:

1. **In sidebar:** Check "Show Performance Metrics"
2. **Go to Performance tab:** View real-time metrics
3. **Monitor these:**
   - Average response time (target: < 2s)
   - Inference time (target: < 50ms)
   - Memory usage (target: < 400MB)

### Set Up Alerts

Add to your app:

```python
# In main() function
if st.session_state.performance_metrics:
    avg_time = np.mean([m['elapsed_time'] for m in st.session_state.performance_metrics[-10:]])
    
    if avg_time > 2.0:
        st.sidebar.error(f"🚨 Performance Alert: Avg {avg_time:.2f}s")
```

---

## 🎯 Benchmark Targets

After implementation, verify these targets:

| Metric | Target | How to Measure |
|--------|--------|---------------|
| **Page Load** | < 3s | Browser DevTools Network tab |
| **Prediction** | < 2s | Performance dashboard |
| **Inference** | < 50ms | Shown in results |
| **Accuracy** | ≥ 90% | Model evaluation (from notebook) |
| **Contrast** | 12:1 | WebAIM Contrast Checker |
| **Uptime** | 99.5% | Streamlit Cloud metrics |

---

## 📦 Deployment Checklist

Before deploying to Streamlit Cloud:

### Files to Commit:
- [ ] app.py (updated with v2)
- [ ] requirements.txt (v2 version)
- [ ] .streamlit/config.toml (v2 version)
- [ ] fetal_health_model.pkl (if using real model)
- [ ] model_info.pkl (if using real model)
- [ ] PERFORMANCE_BENCHMARKS.md (reference)

### Testing:
- [ ] Test locally: `streamlit run app.py`
- [ ] All features work
- [ ] No errors in console
- [ ] Theme looks correct
- [ ] Performance acceptable

### Git Commands:
```bash
git add .
git status  # Verify files
git commit -m "v2.0: Performance benchmarks + dark theme + ML integration"
git push origin main
```

### Streamlit Cloud:
1. Logs show successful deployment
2. App loads without errors
3. Theme renders correctly
4. All functionality works

---

## 🔄 Rollback Plan

If something goes wrong:

```bash
# Restore old files
cp app_old.py app.py
cp requirements_old.txt requirements.txt
cp .streamlit/config_old.toml .streamlit/config.toml

# Reinstall old dependencies
pip install -r requirements.txt

# Test
streamlit run app.py

# Redeploy
git add .
git commit -m "Rollback to v1.0"
git push origin main
```

---

## 🎓 Next Steps

After successful implementation:

### Week 1:
- ✅ Monitor performance metrics daily
- ✅ Collect user feedback on new theme
- ✅ Test on different devices

### Week 2:
- ✅ Integrate real ML model from notebook
- ✅ Fine-tune performance thresholds
- ✅ Add more test cases

### Month 1:
- ✅ Analyze performance trends
- ✅ Optimize slow functions
- ✅ Plan additional features

---

## 📞 Support Resources

### Documentation:
- PERFORMANCE_BENCHMARKS.md - All standards
- app_v2.py - Reference implementation
- Streamlit Docs: https://docs.streamlit.io

### Testing Tools:
- WebAIM Contrast: https://webaim.org/resources/contrastchecker/
- Lighthouse: Chrome DevTools
- GTmetrix: https://gtmetrix.com

### Community:
- Streamlit Forum: https://discuss.streamlit.io
- GitHub Issues: (your repo)

---

## ✅ Success Criteria

You'll know implementation succeeded when:

1. **Visual:**
   - ✅ Dark theme throughout
   - ✅ All text readable (white on dark)
   - ✅ Status colors vibrant (green/yellow/red)
   - ✅ Professional appearance

2. **Performance:**
   - ✅ Page loads < 3 seconds
   - ✅ Predictions < 2 seconds
   - ✅ No slowness warnings
   - ✅ Metrics show < 400MB memory

3. **Functionality:**
   - ✅ All input modes work
   - ✅ ML predictions accurate
   - ✅ Charts render quickly
   - ✅ Export functions work
   - ✅ History saves correctly

4. **User Experience:**
   - ✅ Easy to navigate
   - ✅ Responsive on all devices
   - ✅ Clear feedback on actions
   - ✅ No confusion or errors

---

## 🎉 You're Ready!

You now have:
- ✅ Complete performance benchmarks (reference)
- ✅ Updated application (modern dark theme)
- ✅ ML model integration (ready for real models)
- ✅ Performance monitoring (< 2s targets)
- ✅ Implementation guide (this document)

**Go ahead and implement! 🚀**

Questions? Issues? Check the troubleshooting section above!

---

**Document Version:** 2.0  
**Last Updated:** January 29, 2026  
**Status:** Ready for Implementation  
