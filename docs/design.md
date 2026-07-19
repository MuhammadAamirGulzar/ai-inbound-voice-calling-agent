# OrderSaathi Design System

## Overview
OrderSaathi is a B2B SaaS dashboard for Pakistani restaurant owners to manage an AI voice-agent that handles customer phone calls and takes orders. The design should feel confident, modern, and trustworthy, with clear layouts suitable for busy restaurant environments.

## Visual Identity & Tokens

### Colors
- **Primary Accent**: Confident Violet-Blue (#4F46E5) – conveys trust and modern AI tech.
- **Background (Light)**: Clean neutral off-white (#F9FAFB)
- **Surface**: White (#FFFFFF) with subtle drop shadows
- **Text Primary**: Dark Gray (#111827)
- **Text Secondary**: Medium Gray (#6B7280)
- **Status Colors**:
  - Success/Completed: Emerald Green (#10B981)
  - Warning/Pending: Amber (#F59E0B)
  - Danger/Missed: Rose Red (#F43F5E)
  - Info/In Progress: Sky Blue (#0EA5E9)

### Typography
- **Font Family**: 'Inter', or similar modern sans-serif.
- **Sizes**:
  - Page Titles: 24px/30px bold
  - Section Headers: 18px/24px semibold
  - Body Text: 14px/20px regular
  - Small Text/Labels: 12px/16px medium

### Layout & Spacing
- **Corners**: Rounded corners across components (8px for buttons/inputs, 12px for cards).
- **Whitespace**: Generous padding (16px to 24px) to ensure a clean, uncluttered feel.
- **Navigation**: Left sidebar navigation (fixed, icon + label).
- **Top Bar**: Shows active restaurant name and a prominent call-status indicator ("Agent: Live" / "Agent: Paused").

## Component Library

### 1. KPI / Stat Card
- A clean white card (12px rounded corners, subtle shadow).
- Contains a large numeric value, a descriptive label below or above, and a small colored trend indicator (e.g., "+5% ▲").

### 2. Status Badge / Chip
- Small pill-shaped badges (fully rounded or 8px rounded).
- Background with 10-15% opacity of the status color, text in 100% status color.
- States: "Completed", "Missed", "In Progress", "Pending".

### 3. Data Table
- Clean table with a transparent header row (light gray text, uppercase, sortable arrows).
- White rows with bottom borders.
- Hover state: Very light gray/blue background on row hover.
- Footer with simple pagination controls.

### 4. Audio Player
- Compact component for call recordings.
- Play/Pause toggle button (circular).
- Simple scrubber/waveform bar.
- Timestamps (current time / total time).

### 5. Transcript Viewer (Chat-bubble style)
- Container with light gray background.
- **Agent Turns**: Violet-blue background, white text, aligned to the left.
- **Caller Turns**: White background, dark text, aligned to the right.
- Small timestamp beside or below each bubble.

### 6. Inputs & Controls
- **Buttons**:
  - Primary: Solid violet-blue background, white text, 8px rounded.
  - Secondary: White background, gray border, dark text, 8px rounded.
- **Form Inputs**: 1px gray border, white background, focus state gets a violet-blue ring.
- **Toggle Switches**: Smooth pill toggles for on/off states (like "Agent Active").
