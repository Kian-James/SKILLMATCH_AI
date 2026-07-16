// lib/screens/splash_screen.dart
// In-app splash screen shown while Firebase auth state resolves.
// Full-bleed "poster" treatment: bold brand-color background,
// oversized wordmark, layered decorative shapes.

import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../theme.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  // Entrance animation — plays once.
  late final AnimationController _entryCtrl;
  late final Animation<double> _bgFade;
  late final Animation<double> _shapesFade;
  late final Animation<double> _markFade;
  late final Animation<double> _markScale;
  late final Animation<double> _wordmarkFade;
  late final Animation<Offset> _wordmarkSlide;
  late final Animation<double> _taglineFade;
  late final Animation<double> _loaderFade;

  // Idle loop — subtle continuous motion so the poster feels alive
  // while Firebase resolves, however long that takes.
  late final AnimationController _idleCtrl;

  @override
  void initState() {
    super.initState();

    _entryCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1100),
    )..forward();

    _bgFade = CurvedAnimation(
      parent: _entryCtrl,
      curve: const Interval(0.0, 0.35, curve: Curves.easeOut),
    );
    _shapesFade = CurvedAnimation(
      parent: _entryCtrl,
      curve: const Interval(0.1, 0.55, curve: Curves.easeOut),
    );

    _markFade = CurvedAnimation(
      parent: _entryCtrl,
      curve: const Interval(0.15, 0.55, curve: Curves.easeOut),
    );
    _markScale = Tween<double>(begin: 0.75, end: 1.0).animate(
      CurvedAnimation(
        parent: _entryCtrl,
        curve: const Interval(0.15, 0.65, curve: Curves.easeOutBack),
      ),
    );

    _wordmarkFade = CurvedAnimation(
      parent: _entryCtrl,
      curve: const Interval(0.4, 0.78, curve: Curves.easeOut),
    );
    _wordmarkSlide = Tween<Offset>(
      begin: const Offset(0, 0.3),
      end: Offset.zero,
    ).animate(
      CurvedAnimation(
        parent: _entryCtrl,
        curve: const Interval(0.4, 0.78, curve: Curves.easeOut),
      ),
    );

    _taglineFade = CurvedAnimation(
      parent: _entryCtrl,
      curve: const Interval(0.55, 0.9, curve: Curves.easeOut),
    );

    _loaderFade = CurvedAnimation(
      parent: _entryCtrl,
      curve: const Interval(0.75, 1.0, curve: Curves.easeOut),
    );

    // Slow ambient drift for the background shapes + loader pulse —
    // starts once the entrance settles, loops for as long as needed.
    _idleCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 3200),
    );
    _entryCtrl.addStatusListener((status) {
      if (status == AnimationStatus.completed) {
        _idleCtrl.repeat();
      }
    });
  }

  @override
  void dispose() {
    _entryCtrl.dispose();
    _idleCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    final size = MediaQuery.of(context).size;

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          // ── Full-bleed brand-color background ─────────────────────
          FadeTransition(
            opacity: _bgFade,
            child: DecoratedBox(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    c.brand,
                    Color.lerp(c.brand, Colors.black, 0.28)!,
                  ],
                ),
              ),
            ),
          ),

          // ── Decorative drifting shapes for poster depth ───────────
          FadeTransition(
            opacity: _shapesFade,
            child: AnimatedBuilder(
              animation: _idleCtrl,
              builder: (context, _) {
                final t = _idleCtrl.value * 2 * math.pi;
                return Stack(
                  children: [
                    Positioned(
                      top: -size.width * 0.28 + math.sin(t) * 10,
                      right: -size.width * 0.32,
                      child: _posterCircle(
                        size.width * 0.85,
                        Colors.white.withOpacity(0.07),
                      ),
                    ),
                    Positioned(
                      bottom: -size.width * 0.35,
                      left: -size.width * 0.4 + math.cos(t) * 8,
                      child: _posterCircle(
                        size.width * 0.95,
                        Colors.white.withOpacity(0.06),
                      ),
                    ),
                    Positioned(
                      top: size.height * 0.14,
                      left: -size.width * 0.18,
                      child: _posterCircle(
                        size.width * 0.4,
                        Colors.white.withOpacity(0.05),
                      ),
                    ),
                  ],
                );
              },
            ),
          ),

          // ── Foreground content ─────────────────────────────────────
          SafeArea(
            child: Column(
              children: [
                const Spacer(flex: 4),

                // Icon mark
                FadeTransition(
                  opacity: _markFade,
                  child: ScaleTransition(
                    scale: _markScale,
                    child: Container(
                      width: 132,
                      height: 132,
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(32),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.18),
                            blurRadius: 36,
                            offset: const Offset(0, 16),
                          ),
                        ],
                      ),
                      padding: const EdgeInsets.all(14),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(22),
                        child: Image.asset(
                          'assets/skillmatch_icon_1024.png',
                          fit: BoxFit.cover,
                        ),
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 36),

                // Oversized wordmark — the poster headline
                FadeTransition(
                  opacity: _wordmarkFade,
                  child: SlideTransition(
                    position: _wordmarkSlide,
                    child: Text(
                      'SkillMatch',
                      textAlign: TextAlign.center,
                      style: GoogleFonts.fraunces(
                        fontSize: 46,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                        letterSpacing: -1.0,
                        height: 1.0,
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 14),

                FadeTransition(
                  opacity: _taglineFade,
                  child: Text(
                    'KNOW WHAT TO LEARN NEXT',
                    textAlign: TextAlign.center,
                    style: GoogleFonts.workSans(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: Colors.white.withOpacity(0.85),
                      letterSpacing: 3.2,
                    ),
                  ),
                ),

                const Spacer(flex: 5),

                // Loader — three pulsing dots, poster-appropriate and
                // calmer than a spinner
                FadeTransition(
                  opacity: _loaderFade,
                  child: AnimatedBuilder(
                    animation: _idleCtrl,
                    builder: (context, _) {
                      return Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: List.generate(3, (i) {
                          final phase =
                              (_idleCtrl.value * 3 - i) % 1.0;
                          final wave = (math.sin(phase * 2 * math.pi) + 1) / 2;
                          return Padding(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 5,
                            ),
                            child: Opacity(
                              opacity: 0.35 + wave * 0.65,
                              child: Container(
                                width: 8,
                                height: 8,
                                decoration: const BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: Colors.white,
                                ),
                              ),
                            ),
                          );
                        }),
                      );
                    },
                  ),
                ),

                const SizedBox(height: 48),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _posterCircle(double diameter, Color color) {
    return Container(
      width: diameter,
      height: diameter,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: color,
      ),
    );
  }
}