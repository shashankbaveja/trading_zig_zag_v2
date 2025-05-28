# ZigZag Pattern Recognition Trading Strategy Breakdown

## Section 1: Basic Setup and Inputs

```pinescript
//@version=1
strategy(title='[STRATEGY][RS]ZigZag PA Strategy V4.1', shorttitle='S', overlay=true, pyramiding=0, initial_capital=300000)
```

This sets up the strategy with:
- **overlay=true**: Charts appear on top of price data
- **pyramiding=0**: No adding to existing positions
- **initial_capital=300000**: Starting with $300,000 for backtesting

### User Input Controls
```pinescript
useHA = input(false, title='Use Heikken Ashi Candles')
useAltTF = input(true, title='Use Alt Timeframe')
tf = input('60', title='Alt Timeframe')
showPatterns = input(true, title='Show Patterns')
```

These create toggles that traders can adjust:
- **Heiken Ashi**: Alternative candle type that smooths price action
- **Alt Timeframe**: Uses a different timeframe (60 minutes by default) for analysis
- **Show Patterns**: Whether to display pattern labels on the chart

### Fibonacci Level Toggles
```pinescript
showFib0000 = input(title='Display Fibonacci 0.000:', type=bool, defval=true)
showFib0236 = input(title='Display Fibonacci 0.236:', type=bool, defval=true)
// ... more Fibonacci levels
```

Fibonacci retracements are key support/resistance levels. Each toggle controls whether to show lines at:
- 0% (0.000), 23.6% (0.236), 38.2% (0.382), 50% (0.500)
- 61.8% (0.618), 76.4% (0.764), 100% (1.000)

## Section 2: ZigZag Calculation

```pinescript
zigzag() =>
    _isUp = close >= open
    _isDown = close <= open
    _direction = _isUp[1] and _isDown ? -1 : _isDown[1] and _isUp ? 1 : nz(_direction[1])
    _zigzag = _isUp[1] and _isDown and _direction[1] != -1 ? highest(2) : _isDown[1] and _isUp and _direction[1] != 1 ? lowest(2) : na
```

**What ZigZag does:**
- Identifies swing highs and lows in price movement
- Creates a line that connects significant turning points
- Filters out small price movements to show the main trend

**How it works:**
1. Determines if current candle is up (green) or down (red)
2. Detects direction changes from up to down or vice versa
3. Marks the highest point when trend changes from up to down
4. Marks the lowest point when trend changes from down to up

## Section 3: Pattern Recognition Points (Detailed Explanation)

```pinescript
x = valuewhen(sz, sz, 4) 
a = valuewhen(sz, sz, 3) 
b = valuewhen(sz, sz, 2) 
c = valuewhen(sz, sz, 1) 
d = valuewhen(sz, sz, 0)
```

### What These Lines Do:
These create 5 reference points by looking backwards through the ZigZag turning points:
- **D (sz, 0)**: The most recent ZigZag point (current swing high/low)
- **C (sz, 1)**: The previous ZigZag point (1 swing back)
- **B (sz, 2)**: Two swings back
- **A (sz, 3)**: Three swings back  
- **X (sz, 4)**: Four swings back (oldest reference point)

### Visual Example:
Imagine price action that looks like this over time:
```
     B (high)
    /  \
   /    \
  /      C (low)
 /        \
X (low)    \
            D (high - most recent)
           /
    A (high)
```

### The `valuewhen()` Function Explained:
`valuewhen(condition, source, occurrence)`
- **condition**: `sz` - when a ZigZag point exists (not NA)
- **source**: `sz` - the actual ZigZag value at that point
- **occurrence**: How many instances back to look (0=most recent, 1=previous, etc.)

So `valuewhen(sz, sz, 1)` means: "Give me the ZigZag value from 1 occurrence ago when a ZigZag point existed"

### Ratio Calculations (The Key to Pattern Recognition):
```pinescript
xab = (abs(b-a)/abs(x-a))  // What percentage is the AB move compared to XA move?
xad = (abs(a-d)/abs(x-a))  // What percentage is the AD move compared to XA move?
abc = (abs(b-c)/abs(a-b))  // What percentage is the BC move compared to AB move?
bcd = (abs(c-d)/abs(b-c))  // What percentage is the CD move compared to BC move?
```

### Why These Ratios Matter:
Each harmonic pattern has specific ratio requirements. For example:

**Gartley Pattern Requirements:**
- XAB ratio should be between 0.5 and 0.618 (50% to 61.8%)
- ABC ratio should be between 0.382 and 0.886 (38.2% to 88.6%)
- BCD ratio should be between 1.13 and 2.618 (113% to 261.8%)
- XAD ratio should be between 0.75 and 0.875 (75% to 87.5%)

### Real-World Example:
Let's say we have these price points:
- X = $100
- A = $120  
- B = $110
- C = $115
- D = $108

The ratios would be:
- **XAB** = |110-120|/|100-120| = 10/20 = 0.5 (50%)
- **ABC** = |115-110|/|120-110| = 5/10 = 0.5 (50%)
- **BCD** = |108-115|/|110-115| = 7/5 = 1.4 (140%)
- **XAD** = |120-108|/|100-120| = 12/20 = 0.6 (60%)

If these ratios match a known pattern's requirements, the algorithm recognizes it as that pattern.

### Why This Approach Works:
1. **Mathematical Precision**: Instead of subjective "this looks like a pattern," it uses exact measurements
2. **Historical Validation**: These ratios are based on Fibonacci numbers and have shown statistical significance in markets
3. **Objective Detection**: Removes human bias and emotion from pattern recognition
4. **Consistent Application**: The same mathematical rules apply regardless of timeframe or market

### The Five-Point System:
Most harmonic patterns use 5 points because:
- **Minimum Complexity**: Enough points to create meaningful geometric relationships
- **Maximum Clarity**: Not so many points that patterns become overly complex
- **Historical Precedent**: Developed by traders like Scott Carney and Larry Pesavento through decades of market analysis
- **Fibonacci Foundation**: The ratios often align with Fibonacci retracement levels (0.236, 0.382, 0.618, etc.)

## Section 4: Harmonic Pattern Functions

The strategy recognizes 17 different harmonic patterns, each with specific ratio requirements:

### Classic Patterns:
- **Bat Pattern**: XAB (38.2%-50%), ABC (38.2%-88.6%), BCD (161.8%-261.8%)
- **Gartley Pattern**: XAB (50%-61.8%), ABC (38.2%-88.6%), BCD (113%-261.8%)
- **Butterfly Pattern**: XAB (≤78.6%), ABC (38.2%-88.6%), BCD (161.8%-261.8%)
- **Crab Pattern**: XAB (50%-87.5%), ABC (38.2%-88.6%), BCD (200%-500%)

### Other Patterns:
- **ABCD**: Simple 3-point pattern
- **Shark, Wolf Wave, Head & Shoulders**: More complex formations
- **Contracting/Expanding Triangles**: Consolidation patterns
- **Anti-patterns**: Inverse versions of classic patterns

Each pattern function checks if the current price ratios match the specific requirements for that pattern.

## Section 5: Pattern Visualization

```pinescript
plotshape(not showPatterns ? na : isABCD(-1) and not isABCD(-1)[1], text="\nAB=CD", title='Bear ABCD', style=shape.labeldown, color=maroon, textcolor=white, location=location.top, transp=0, offset=-2)
```

This section displays labels on the chart when patterns are detected:
- **Bear patterns** (selling opportunities): Red labels above price
- **Bull patterns** (buying opportunities): Green labels below price
- Only shows new patterns (not repeating from previous bar)

## Section 6: Fibonacci Retracement Lines

```pinescript
fib_range = abs(d-c)
fib_0000 = not showFib0000 ? na : d > c ? d-(fib_range*0.000):d+(fib_range*0.000)
// ... more Fibonacci calculations
```

**Purpose:** Creates horizontal support/resistance lines based on the most recent price swing (from point C to point D).

**How it works:**
- Calculates the distance between points C and D
- Projects Fibonacci retracement levels from point D
- Adjusts direction based on whether D is above or below C

## Section 7: Trading Logic

### Trade Setup Parameters:
```pinescript
target01_trade_size = input(title='Target 1 - Trade size:', type=float, defval=10000.00)
target01_ew_rate = input(title='Target 1 - Fib. Rate to use for Entry Window:', type=float, defval=0.236)
target01_tp_rate = input(title='Target 1 - Fib. Rate to use for TP:', type=float, defval=0.618)
target01_sl_rate = input(title='Target 1 - Fib. Rate to use for SL:', type=float, defval=-0.236)
```

**Entry Window (EW)**: Price level where trades are allowed to enter
**Take Profit (TP)**: Price level to close profitable trades
**Stop Loss (SL)**: Price level to close losing trades

### Entry and Exit Conditions:
```pinescript
target01_buy_entry = (buy_patterns_00 or buy_patterns_01) and close <= f_last_fib(target01_ew_rate)
target01_buy_close = high >= f_last_fib(target01_tp_rate) or low <= f_last_fib(target01_sl_rate)
```

**Buy Entry**: Pattern detected AND price is at/below entry window
**Buy Exit**: Price hits take profit OR stop loss

## How the Strategy Works (Summary)

### 1. **Pattern Detection Phase**
- ZigZag identifies significant price swings
- Algorithm continuously monitors for 17 different harmonic patterns
- Each pattern has specific mathematical ratios that must be met

### 2. **Setup Phase**
- When a pattern completes, Fibonacci retracement levels are calculated
- These levels act as potential support/resistance zones
- Entry window, take profit, and stop loss levels are set

### 3. **Entry Phase**
- **For Bull Patterns**: Enter long when price drops to entry window (23.6% Fibonacci level by default)
- **For Bear Patterns**: Enter short when price rises to entry window
- This allows entering at better prices after pattern completion

### 4. **Exit Phase**
- **Take Profit**: Close at favorable Fibonacci level (61.8% by default)
- **Stop Loss**: Close if price moves against the trade (-23.6% by default)

### 5. **Key Strategy Principles**
- **Harmonic patterns** suggest probable reversal points
- **Fibonacci levels** provide precise entry and exit targets
- **Multiple timeframe analysis** (optional) for better accuracy
- **Risk management** through predetermined stop losses

### 6. **Two-Target System**
- **Target 1**: Conservative approach with closer take profit
- **Target 2**: (Optional) More aggressive approach with extended targets
- Allows for scaling out of positions or different risk profiles

## Why This Strategy Can Be Effective

1. **Mathematical Precision**: Uses specific ratios rather than subjective analysis
2. **Multiple Confirmation**: Combines pattern recognition with Fibonacci levels
3. **Built-in Risk Management**: Predetermined entry, exit, and stop loss levels
4. **Versatility**: Works on both bullish and bearish setups
5. **Backtesting Capability**: Can be tested on historical data for performance evaluation

The strategy essentially tries to catch price reversals at mathematically significant levels, using time-tested harmonic patterns that have shown statistical edge in financial markets.