// lib/widgets/tutorial_overlay.dart
//
// Adobe Scan-style spotlight tutorial.
// Usage:
//   TutorialOverlay(steps: [...], onFinish: () {})
//
// Each step targets a GlobalKey so the overlay spotlights the widget.

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../theme.dart';

// ── Step definition ───────────────────────────────────────────────────────────

class TutorialStep {
  final GlobalKey targetKey; // widget to spotlight
  final String title;
  final String body;
  final SpotlightShape shape; // circle or roundedRect

  const TutorialStep({
    required this.targetKey,
    required this.title,
    required this.body,
    this.shape = SpotlightShape.roundedRect,
  });
}

enum SpotlightShape { circle, roundedRect }

// ── Overlay widget ────────────────────────────────────────────────────────────

class TutorialOverlay extends StatefulWidget {
  final List<TutorialStep> steps;
  final VoidCallback onFinish;

  const TutorialOverlay({
    super.key,
    required this.steps,
    required this.onFinish,
  });

  @override
  State<TutorialOverlay> createState() => _TutorialOverlayState();
}

class _TutorialOverlayState extends State<TutorialOverlay>
    with TickerProviderStateMixin {
  int _current = 0;

  // spotlight animation
  late AnimationController _spotCtrl;
  late Animation<double> _spotAnim;

  // card slide-up animation
  late AnimationController _cardCtrl;
  late Animation<Offset> _cardAnim;
  late Animation<double> _cardFade;

  @override
  void initState() {
    super.initState();

    _spotCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 400));
    _spotAnim = CurvedAnimation(parent: _spotCtrl, curve: Curves.easeOutCubic);

    _cardCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 350));
    _cardAnim = Tween<Offset>(begin: const Offset(0, 0.12), end: Offset.zero)
        .animate(
            CurvedAnimation(parent: _cardCtrl, curve: Curves.easeOutCubic));
    _cardFade = CurvedAnimation(parent: _cardCtrl, curve: Curves.easeOut);

    _spotCtrl.forward();
    _cardCtrl.forward();
  }

  @override
  void dispose() {
    _spotCtrl.dispose();
    _cardCtrl.dispose();
    super.dispose();
  }

  Future<void> _goTo(int index) async {
    await _cardCtrl.reverse();
    setState(() => _current = index);
    _spotCtrl.forward(from: 0);
    _cardCtrl.forward(from: 0);
  }

  void _next() {
    if (_current < widget.steps.length - 1) {
      _goTo(_current + 1);
    } else {
      widget.onFinish();
    }
  }

  void _back() {
    if (_current > 0) _goTo(_current - 1);
  }

  // Get bounding rect of the target widget in global coordinates
  Rect? _targetRect(TutorialStep step) {
    final ctx = step.targetKey.currentContext;
    if (ctx == null) return null;
    final box = ctx.findRenderObject() as RenderBox?;
    if (box == null) return null;
    final pos = box.localToGlobal(Offset.zero);
    return pos & box.size;
  }

  @override
  Widget build(BuildContext context) {
    final step = widget.steps[_current];
    final total = widget.steps.length;
    final c = AppColors.of(context);
    final size = MediaQuery.of(context).size;
    final rect = _targetRect(step);
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Material(
      color: Colors.transparent,
      child: Stack(
        children: [
          // ── Dimmed backdrop with spotlight cutout ─────────────────────
          AnimatedBuilder(
            animation: _spotAnim,
            builder: (_, __) {
              return CustomPaint(
                size: size,
                painter: _SpotlightPainter(
                  rect: rect,
                  shape: step.shape,
                  progress: _spotAnim.value,
                  dimColor: Colors.black.withOpacity(isDark ? 0.72 : 0.55),
                ),
              );
            },
          ),

          // ── Tap outside → next ────────────────────────────────────────
          Positioned.fill(
            child: GestureDetector(
              behavior: HitTestBehavior.translucent,
              onTap: _next,
            ),
          ),

          // ── Step card ─────────────────────────────────────────────────
          _buildCard(context, step, total, c, size, rect),
        ],
      ),
    );
  }

  Widget _buildCard(BuildContext context, TutorialStep step, int total,
      AppColors c, Size size, Rect? rect) {
    // Decide if card goes above or below the spotlight
    final double screenMid = size.height / 2;
    final double rectMid = rect != null ? rect.center.dy : screenMid;
    final bool cardBelow = rectMid < screenMid;

    final double cardTop = cardBelow
        ? (rect != null ? rect.bottom + 20 : screenMid + 60)
        : null as double? ?? 0;
    final double cardBottom = !cardBelow
        ? (rect != null ? size.height - rect.top + 20 : size.height / 2 - 60)
        : null as double? ?? 0;

    return Positioned(
      left: 20,
      right: 20,
      top: cardBelow ? cardTop.clamp(60, size.height - 280) : null,
      bottom: !cardBelow ? cardBottom.clamp(60, size.height - 280) : null,
      child: SlideTransition(
        position: _cardAnim,
        child: FadeTransition(
          opacity: _cardFade,
          child: Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: c.navy800,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: c.brand.withOpacity(0.3), width: 1.5),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.35),
                  blurRadius: 32,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Step indicator dots
                Row(children: [
                  ...List.generate(
                      total,
                      (i) => AnimatedContainer(
                            duration: const Duration(milliseconds: 250),
                            margin: const EdgeInsets.only(right: 5),
                            width: i == _current ? 18 : 6,
                            height: 6,
                            decoration: BoxDecoration(
                              color: i == _current
                                  ? c.brand
                                  : c.brand.withOpacity(0.25),
                              borderRadius: BorderRadius.circular(3),
                            ),
                          )),
                  const Spacer(),
                  Text(
                    '${_current + 1} of $total',
                    style:
                        GoogleFonts.inter(fontSize: 11, color: c.textTertiary),
                  ),
                  const SizedBox(width: 10),
                  // Skip button
                  GestureDetector(
                    onTap: widget.onFinish,
                    child: Text(
                      'Skip',
                      style: GoogleFonts.inter(
                          fontSize: 12,
                          color: c.textTertiary,
                          fontWeight: FontWeight.w500,
                          decoration: TextDecoration.underline),
                    ),
                  ),
                ]),
                const SizedBox(height: 14),

                // Title
                Text(step.title,
                    style: GoogleFonts.spaceGrotesk(
                        fontSize: 17,
                        fontWeight: FontWeight.w700,
                        color: c.textPrimary)),
                const SizedBox(height: 8),

                // Body
                Text(step.body,
                    style: GoogleFonts.inter(
                        fontSize: 13, color: c.textSecondary, height: 1.55)),
                const SizedBox(height: 18),

                // Back / Next buttons
                Row(children: [
                  if (_current > 0)
                    Expanded(
                      child: OutlinedButton(
                        onPressed: _back,
                        style: OutlinedButton.styleFrom(
                          foregroundColor: c.textSecondary,
                          side: BorderSide(color: c.cardBorder),
                          padding: const EdgeInsets.symmetric(vertical: 12),
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10)),
                        ),
                        child: Text('Back',
                            style: GoogleFonts.spaceGrotesk(
                                fontWeight: FontWeight.w600)),
                      ),
                    ),
                  if (_current > 0) const SizedBox(width: 10),
                  Expanded(
                    flex: 2,
                    child: ElevatedButton(
                      onPressed: _next,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: c.brand,
                        foregroundColor:
                            Theme.of(context).brightness == Brightness.dark
                                ? AppTheme.navy900
                                : Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(10)),
                        elevation: 0,
                      ),
                      child: Text(
                        _current == total - 1 ? "Let's go!" : 'Next',
                        style: GoogleFonts.spaceGrotesk(
                            fontWeight: FontWeight.w700),
                      ),
                    ),
                  ),
                ]),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ── Spotlight painter ─────────────────────────────────────────────────────────

class _SpotlightPainter extends CustomPainter {
  final Rect? rect;
  final SpotlightShape shape;
  final double progress;
  final Color dimColor;

  const _SpotlightPainter({
    required this.rect,
    required this.shape,
    required this.progress,
    required this.dimColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final dim = Paint()..color = dimColor;

    if (rect == null || progress == 0) {
      canvas.drawRect(Offset.zero & size, dim);
      return;
    }

    // Pad the spotlight slightly
    final pad = 10.0 * progress;
    final padded = rect!.inflate(pad);

    // Animate from center outward
    final animated = Rect.lerp(
      Rect.fromCenter(center: padded.center, width: 0, height: 0),
      padded,
      progress,
    )!;

    canvas.drawRect(Offset.zero & size, dim);

    final clearPaint = Paint()..blendMode = BlendMode.clear;
    canvas.saveLayer(null, Paint());
    canvas.drawRect(Offset.zero & size, dim);

    if (shape == SpotlightShape.circle) {
      canvas.drawCircle(animated.center, animated.shortestSide / 2, clearPaint);
    } else {
      canvas.drawRRect(
          RRect.fromRectAndRadius(animated, const Radius.circular(14)),
          clearPaint);
    }
    canvas.restore();
  }

  @override
  bool shouldRepaint(_SpotlightPainter old) =>
      old.rect != rect || old.progress != progress;
}
