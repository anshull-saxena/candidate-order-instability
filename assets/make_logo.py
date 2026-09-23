"""Generate the Slaya logo: animated SVG sources plus PNG renders.

Slaya: Slaying Choice-Order Instability in Non-Autoregressive Decision Models.
The mark combines:
1. The classic Laya Sanskrit dissolution spiral (the foundation).
2. Slaying energy blade slicing through candidate-order instability.
3. Concentric gyroscopic rings and 5 orbiting candidate permutation nodes (C_K cyclic shifts).
4. Pure SVG + CSS keyframe animations for high-velocity rotation, reactor core pulsing, 
   radar shockwaves, and gradient shimmer.
"""
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(REPO, "assets")
os.makedirs(ASSETS, exist_ok=True)

SVG_CONTENT = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 168" width="100%" height="168" role="img" aria-label="Slaya: Slaying Choice-Order Instability in Decision Models">
  <defs>
    <!-- Background Space Void Gradient -->
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#04060b" />
      <stop offset="50%" stop-color="#0a0f1a" />
      <stop offset="100%" stop-color="#030508" />
    </linearGradient>

    <!-- Glowing Cyber Border Gradient -->
    <linearGradient id="borderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00f0ff" stop-opacity="0.85" />
      <stop offset="25%" stop-color="#3b82f6" stop-opacity="0.35" />
      <stop offset="50%" stop-color="#a855f7" stop-opacity="0.4" />
      <stop offset="75%" stop-color="#ec4899" stop-opacity="0.7" />
      <stop offset="100%" stop-color="#00f0ff" stop-opacity="0.4" />
    </linearGradient>

    <!-- Multi-Tone Energy Gradients -->
    <linearGradient id="cyanPurple" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00f0ff" />
      <stop offset="50%" stop-color="#8b5cf6" />
      <stop offset="100%" stop-color="#ec4899" />
    </linearGradient>

    <linearGradient id="magentaAmber" x1="100%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#ff007f" />
      <stop offset="60%" stop-color="#f59e0b" />
      <stop offset="100%" stop-color="#00f0ff" />
    </linearGradient>

    <!-- Dynamic Slaya Wordmark Liquid Gradient -->
    <linearGradient id="slayaLiquidGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#ffffff">
        <animate attributeName="stop-color" 
          values="#ffffff;#38bdf8;#c084fc;#f43f5e;#38bdf8;#ffffff" 
          dur="5s" repeatCount="indefinite" />
      </stop>
      <stop offset="25%" stop-color="#38bdf8">
        <animate attributeName="stop-color" 
          values="#38bdf8;#c084fc;#f43f5e;#ffffff;#38bdf8;#38bdf8" 
          dur="5s" repeatCount="indefinite" />
      </stop>
      <stop offset="50%" stop-color="#c084fc">
        <animate attributeName="stop-color" 
          values="#c084fc;#f43f5e;#ffffff;#38bdf8;#c084fc;#c084fc" 
          dur="5s" repeatCount="indefinite" />
      </stop>
      <stop offset="75%" stop-color="#f43f5e">
        <animate attributeName="stop-color" 
          values="#f43f5e;#ffffff;#38bdf8;#c084fc;#f43f5e;#f43f5e" 
          dur="5s" repeatCount="indefinite" />
      </stop>
      <stop offset="100%" stop-color="#ffffff">
        <animate attributeName="stop-color" 
          values="#ffffff;#38bdf8;#c084fc;#f43f5e;#38bdf8;#ffffff" 
          dur="5s" repeatCount="indefinite" />
      </stop>
    </linearGradient>

    <!-- Core Radial Glow -->
    <radialGradient id="reactorGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="1" />
      <stop offset="30%" stop-color="#00f0ff" stop-opacity="0.9" />
      <stop offset="65%" stop-color="#8b5cf6" stop-opacity="0.5" />
      <stop offset="100%" stop-color="#04060b" stop-opacity="0" />
    </radialGradient>

    <!-- Ambient Light Auras -->
    <radialGradient id="leftAura" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#00f0ff" stop-opacity="0.22" />
      <stop offset="60%" stop-color="#6366f1" stop-opacity="0.08" />
      <stop offset="100%" stop-color="#04060b" stop-opacity="0" />
    </radialGradient>

    <radialGradient id="rightAura" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#ec4899" stop-opacity="0.16" />
      <stop offset="60%" stop-color="#8b5cf6" stop-opacity="0.06" />
      <stop offset="100%" stop-color="#04060b" stop-opacity="0" />
    </radialGradient>

    <!-- Neon Glow Filter -->
    <filter id="laserGlow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="3.2" result="blur1" />
      <feGaussianBlur stdDeviation="7.5" result="blur2" />
      <feMerge>
        <feMergeNode in="blur2" />
        <feMergeNode in="blur1" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>

    <filter id="textGlow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="4.5" result="glow" />
      <feMerge>
        <feMergeNode in="glow" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>

    <!-- Background Tech Grid Pattern -->
    <pattern id="techGrid" width="24" height="24" patternUnits="userSpaceOnUse">
      <path d="M 24 0 L 0 0 0 24" fill="none" stroke="#1e293b" stroke-width="0.75" opacity="0.25" />
      <circle cx="24" cy="0" r="0.9" fill="#38bdf8" opacity="0.4" />
    </pattern>

    <style>
      @keyframes spinClockwise {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }
      @keyframes spinCounter {
        from { transform: rotate(360deg); }
        to { transform: rotate(0deg); }
      }
      @keyframes pulseReactor {
        0%, 100% {
          transform: scale(0.96);
          opacity: 0.9;
        }
        50% {
          transform: scale(1.15);
          opacity: 1;
        }
      }
      @keyframes radarWave1 {
        0% { r: 10px; opacity: 0.9; stroke-width: 2.2px; }
        100% { r: 56px; opacity: 0; stroke-width: 0.5px; }
      }
      @keyframes radarWave2 {
        0% { r: 10px; opacity: 0.9; stroke-width: 2.2px; }
        100% { r: 56px; opacity: 0; stroke-width: 0.5px; }
      }
      @keyframes liveStatus {
        0%, 100% { opacity: 1; transform: scale(1); filter: drop-shadow(0 0 3px #10b981); }
        50% { opacity: 0.35; transform: scale(0.85); filter: drop-shadow(0 0 0px #10b981); }
      }
      @keyframes bladeGlint {
        0%, 100% { stroke: #00f0ff; filter: drop-shadow(0 0 4px #00f0ff); }
        50% { stroke: #ec4899; filter: drop-shadow(0 0 8px #ec4899); }
      }
      @keyframes textFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-1.5px); }
      }

      .spin-slow {
        transform-origin: 94px 84px;
        animation: spinClockwise 14s linear infinite;
      }
      .spin-medium {
        transform-origin: 94px 84px;
        animation: spinClockwise 7s linear infinite;
      }
      .spin-rev {
        transform-origin: 94px 84px;
        animation: spinCounter 9s linear infinite;
      }
      .spin-nodes {
        transform-origin: 94px 84px;
        animation: spinClockwise 5.5s cubic-bezier(0.4, 0, 0.2, 1) infinite;
      }
      .reactor-core {
        transform-origin: 94px 84px;
        animation: pulseReactor 2.2s ease-in-out infinite;
      }
      .radar-1 {
        animation: radarWave1 2.2s cubic-bezier(0.1, 0.8, 0.3, 1) infinite;
      }
      .radar-2 {
        animation: radarWave2 2.2s cubic-bezier(0.1, 0.8, 0.3, 1) 1.1s infinite;
      }
      .status-pulse {
        transform-origin: 208px 136px;
        animation: liveStatus 1.6s ease-in-out infinite;
      }
      .blade-path {
        animation: bladeGlint 3s ease-in-out infinite;
      }
      .wordmark-group {
        animation: textFloat 4s ease-in-out infinite;
      }
      .wordmark {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", "Arial Black", sans-serif;
        font-weight: 900;
        letter-spacing: 6px;
      }
      .subtitle {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-weight: 700;
        letter-spacing: 3.2px;
      }
      .telemetry {
        font-family: "JetBrains Mono", "SF Mono", Monaco, "Cascadia Code", "Fira Code", monospace;
        font-weight: 600;
        letter-spacing: 0.8px;
      }
    </style>
  </defs>

  <!-- Outer Card Frame with Rounded Corners -->
  <rect x="2" y="2" width="776" height="164" rx="22" fill="url(#bgGrad)" stroke="url(#borderGrad)" stroke-width="1.8" />
  
  <!-- Subtle High-Tech Blueprint Grid Background -->
  <rect x="4" y="4" width="772" height="160" rx="20" fill="url(#techGrid)" />

  <!-- Luminous Ambient Light Halos -->
  <ellipse cx="94" cy="84" rx="95" ry="75" fill="url(#leftAura)" />
  <ellipse cx="500" cy="84" rx="160" ry="75" fill="url(#rightAura)" />

  <!-- ==================== LEFT EMBLEM: THE SLAYA CYCLIC ENGINE ==================== -->
  <g id="cyclic-emblem">
    <!-- Static Outer Gyroscope Calibration Ring -->
    <circle cx="94" cy="84" r="58" fill="none" stroke="#1e293b" stroke-width="1.2" stroke-dasharray="3 7" />
    <circle cx="94" cy="84" r="66" fill="none" stroke="#334155" stroke-width="0.8" opacity="0.5" stroke-dasharray="1 13" />
    
    <!-- Expanding Invariance Radar Waves from Decision Core -->
    <circle cx="94" cy="84" r="10" fill="none" stroke="#00f0ff" class="radar-1" />
    <circle cx="94" cy="84" r="10" fill="none" stroke="#ec4899" class="radar-2" />

    <!-- Outer Fast Ring with Segmented Energy Arcs (Clockwise) -->
    <g class="spin-medium">
      <circle cx="94" cy="84" r="51" fill="none" stroke="url(#cyanPurple)" stroke-width="3" stroke-linecap="round" stroke-dasharray="75 35 20 30" />
      <circle cx="94" cy="33" r="3" fill="#00f0ff" filter="url(#laserGlow)" />
      <circle cx="145" cy="84" r="2.5" fill="#ec4899" filter="url(#laserGlow)" />
    </g>

    <!-- Inner Counter-Rotating Telemetry Ring (Counter-Clockwise) -->
    <g class="spin-rev">
      <circle cx="94" cy="84" r="39" fill="none" stroke="url(#magentaAmber)" stroke-width="2" stroke-linecap="round" stroke-dasharray="32 18 8 22" opacity="0.85" />
      <line x1="94" y1="45" x2="94" y2="50" stroke="#00f0ff" stroke-width="2" />
      <line x1="133" y1="84" x2="128" y2="84" stroke="#00f0ff" stroke-width="2" />
      <line x1="94" y1="123" x2="94" y2="118" stroke="#00f0ff" stroke-width="2" />
      <line x1="55" y1="84" x2="60" y2="84" stroke="#00f0ff" stroke-width="2" />
    </g>

    <!-- Foundation: Iconic Sanskrit Laya Dissolution Arc & Inward Spiral Dots -->
    <g opacity="0.55">
      <path d="M 81.79 61.04 A 26.0 26.0 0 1 0 107.00 106.52" fill="none" stroke="#38bdf8" stroke-width="2.6" stroke-linecap="round" />
      <circle cx="114.98" cy="94.46" r="2.11" fill="#38bdf8" opacity="0.87" />
      <circle cx="114.44" cy="81.49" r="1.88" fill="#38bdf8" opacity="0.82" />
      <circle cx="107.40" cy="72.56" r="1.66" fill="#38bdf8" opacity="0.76" />
      <circle cx="98.01" cy="70.00" r="1.46" fill="#38bdf8" opacity="0.69" />
      <circle cx="90.56" cy="73.08" r="1.26" fill="#38bdf8" opacity="0.62" />
      <circle cx="87.57" cy="78.79" r="1.07" fill="#38bdf8" opacity="0.53" />
      <circle cx="88.96" cy="83.51" r="0.88" fill="#38bdf8" opacity="0.44" />
      <circle cx="92.39" cy="84.85" r="0.70" fill="#38bdf8" opacity="0.35" />
    </g>

    <!-- 5 Orbiting Candidate Nodes: Cyclic Permutation Shift C_K -->
    <g class="spin-nodes">
      <!-- Candidate 0: Cyan -->
      <circle cx="94" cy="37" r="4.2" fill="#00f0ff" filter="url(#laserGlow)" />
      <!-- Candidate 1: Magenta -->
      <circle cx="138" cy="69" r="3.8" fill="#ec4899" filter="url(#laserGlow)" />
      <!-- Candidate 2: Amber -->
      <circle cx="121" cy="122" r="3.8" fill="#fbbf24" filter="url(#laserGlow)" />
      <!-- Candidate 3: Emerald -->
      <circle cx="67" cy="122" r="3.8" fill="#10b981" filter="url(#laserGlow)" />
      <!-- Candidate 4: Violet -->
      <circle cx="50" cy="69" r="3.8" fill="#8b5cf6" filter="url(#laserGlow)" />
    </g>

    <!-- The Slaying Energy Blade ("Slaya" Slicing Through Order Instability) -->
    <g>
      <path d="M 64 114 C 74 100, 84 90, 94 84 C 110 75, 124 64, 132 46" fill="none" stroke="url(#cyanPurple)" stroke-width="4.2" stroke-linecap="round" class="blade-path" />
      <!-- Sharp Blade Tips -->
      <polygon points="60,119 66,110 68,116" fill="#00f0ff" />
      <polygon points="130,44 136,52 128,50" fill="#ec4899" />
    </g>

    <!-- Central Invariant Reactor Core -->
    <g class="reactor-core">
      <circle cx="94" cy="84" r="14" fill="url(#reactorGlow)" />
      <circle cx="94" cy="84" r="7.5" fill="#ffffff" filter="url(#laserGlow)" />
      <circle cx="94" cy="84" r="3.2" fill="#0b101c" />
      <circle cx="94" cy="84" r="1.4" fill="#00f0ff" />
    </g>
  </g>

  <!-- ==================== CENTER & RIGHT: WORDMARK & TELEMETRY ==================== -->
  <g transform="translate(194, 0)">
    <!-- Wordmark Group with Gentle Float Animation -->
    <g class="wordmark-group">
      <!-- Glow Drop Shadow Behind SLAYA -->
      <text x="0" y="82" class="wordmark" font-size="62" fill="url(#slayaLiquidGrad)" filter="url(#textGlow)">SLAYA</text>
      <!-- Crisp Foreground SLAYA -->
      <text x="0" y="82" class="wordmark" font-size="62" fill="url(#slayaLiquidGrad)">SLAYA</text>
    </g>

    <!-- Top Badge: C_K INVARIANT ENGINE -->
    <g transform="translate(290, 36)">
      <rect x="0" y="0" width="138" height="22" rx="11" fill="#111827" stroke="#38bdf8" stroke-width="1.2" opacity="0.9" />
      <circle cx="12" cy="11" r="3" fill="#00f0ff" filter="url(#laserGlow)" />
      <text x="76" y="15" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="9.5" font-weight="800" fill="#38bdf8" text-anchor="middle" letter-spacing="1.2">C_K INVARIANT</text>
    </g>

    <!-- Subtitle: Expanded Description -->
    <text x="2" y="108" class="subtitle" font-size="11.5" fill="#94a3b8">
      <tspan fill="#38bdf8">CYCLIC</tspan> DECISION ENGINE <tspan fill="#475569">│</tspan> <tspan fill="#e2e8f0" font-weight="600">ZERO CHOICE-ORDER BIAS</tspan>
    </text>

    <!-- Bottom Telemetry HUD Pill -->
    <g transform="translate(0, 122)">
      <!-- Pill Container -->
      <rect x="0" y="0" width="554" height="28" rx="14" fill="#080c14" stroke="#1e293b" stroke-width="1.2" />
      
      <!-- Pulsing Live Radar Indicator Dot -->
      <circle cx="16" cy="14" r="5" fill="#10b981" class="status-pulse" />
      <circle cx="16" cy="14" r="2.2" fill="#ffffff" />
      
      <!-- Telemetry Readings -->
      <text x="30" y="18" class="telemetry" font-size="10.5" fill="#cbd5e1">
        <tspan fill="#10b981" font-weight="700">FLIP RATE: 0.0%</tspan>
        <tspan fill="#475569"> │ </tspan>
        <tspan fill="#38bdf8">M=5 CYCLIC SHIFT</tspan>
        <tspan fill="#475569"> │ </tspan>
        <tspan fill="#f43f5e" font-weight="600">VANILLA LAYA: 47.3%</tspan>
        <tspan fill="#475569"> │ </tspan>
        <tspan fill="#c084fc">JEV: 0% AUDIT</tspan>
      </text>

      <!-- Right Diamond Particle -->
      <polygon points="536,14 540,8 544,14 540,20" fill="#00f0ff" opacity="0.85" filter="url(#laserGlow)" />
    </g>
  </g>
</svg>
'''

def main():
    svg_path = os.path.join(ASSETS, "slaya-logo-animated.svg")
    with open(svg_path, "w") as f:
        f.write(SVG_CONTENT.strip())
    print(f"Wrote animated SVG to {svg_path} ({os.path.getsize(svg_path)} bytes)")

    png_path = os.path.join(ASSETS, "slaya-logo.png")
    subprocess.run(["rsvg-convert", "-w", "1560", "-a", "-o", png_path, svg_path], check=True)
    print(f"Rendered PNG fallback to {png_path} ({os.path.getsize(png_path)} bytes)")

if __name__ == "__main__":
    main()
