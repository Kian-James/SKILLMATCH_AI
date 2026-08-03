// lib/theme.dart
// Direction: Neo-brutalism + bento grid.
// Thick black outlines, hard offset shadows, flat saturated color blocks,
// chunky-but-rounded corners. No gradients, no soft blur shadows.

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  // ── Ink (the "brutal" outline/text color in both modes) ───────────────────
  static const Color ink = Color(0xFF14150F); // near-black, warm
  static const Color paper = Color(0xFFF3F1E7); // warm off-white page bg
  static const Color paperCard = Color(0xFFFFFFFF);

  // ── Dark surface stack ──────────────────────────────────────────────────
  static const Color navy900 = Color(0xFF121310);
  static const Color navy800 = Color(0xFF1B1C17);
  static const Color navy700 = Color(0xFF24261F);
  static const Color navy600 = Color(0xFF2E3028);
  static const Color navy500 = Color(0xFF3D4036);
  static const Color navy400 = Color(0xFF54584A);

  // ── Light surface stack ─────────────────────────────────────────────────
  static const Color light50 = Color(0xFFF3F1E7);
  static const Color light100 = Color(0xFFEAE6D6);
  static const Color light200 = Color(0xFFDCD6BE);
  static const Color light300 = Color(0xFFC6BE9C);
  static const Color light400 = Color(0xFF9C9473);

  // ── Brand — punchy bento green ──────────────────────────────────────────
  static const Color teal500 = Color(0xFF3DDC84); // bright block green
  static const Color teal400 = Color(0xFF6BE39F);
  static const Color teal300 = Color(0xFF9AEDBB);
  static const Color teal200 = Color(0xFFC6F5D8);
  static const Color teal100 = Color(0xFFE4FAEC);
  static const Color tealDark = Color(0xFF0E6B3A); // light-mode text-on-tint

  // Gold — credential / warning block
  static const Color amber500 = Color(0xFFFFB627);
  static const Color amber400 = Color(0xFFFFC85C);
  // Coral — danger block
  static const Color red500 = Color(0xFFFF5A45);
  static const Color red400 = Color(0xFFFF8672);
  // Sky — info / matched-secondary block
  static const Color green500 = Color(0xFF2FB88A);
  static const Color green400 = Color(0xFF5FCBA3);

  // ── Dark semantic tokens ───────────────────────────────────────────────
  static const Color textPrimary = Color(0xFFF3F1E7);
  static const Color textSecondary = Color(0xFFB7B7A6);
  static const Color textTertiary = Color(0xFF7C7C6C);
  static const Color cardBorder = Color(0xFFF3F1E7); // outline pops in dark
  static const Color divider = Color(0xFF2E3028);

  // ── Light semantic tokens ──────────────────────────────────────────────
  static const Color ltTextPrimary = Color(0xFF14150F);
  static const Color ltTextSecondary = Color(0xFF4B4C3F);
  static const Color ltTextTertiary = Color(0xFF848264);
  static const Color ltCardBorder = Color(0xFF14150F); // solid ink outline
  static const Color ltDivider = Color(0xFFDCD6BE);
  static const Color ltCardBg = Color(0xFFFFFFFF);
  static const Color ltSurface = Color(0xFFF3F1E7);

  // ── Brutalist constants ─────────────────────────────────────────────────
  static const double borderWidth = 2.5;
  static const double borderWidthThick = 3;
  static const double radius = 14;
  static const double radiusSm = 10;

  /// Hard offset shadow — no blur, flat block of color behind the card.
  /// This is the signature neo-brutalist "sticker" effect.
  static List<BoxShadow> hardShadow(Color color,
          {double dx = 5, double dy = 5}) =>
      [BoxShadow(color: color, offset: Offset(dx, dy), blurRadius: 0)];

  static BoxDecoration neoBox({
    required BuildContext context,
    Color? fill,
    Color? shadowColor,
    double radius = AppTheme.radius,
    double borderWidthOverride = borderWidth,
    double dx = 5,
    double dy = 5,
  }) {
    final c = AppColors.of(context);
    return BoxDecoration(
      color: fill ??
          (Theme.of(context).brightness == Brightness.dark
              ? navy800
              : ltCardBg),
      borderRadius: BorderRadius.circular(radius),
      border: Border.all(color: c.cardBorder, width: borderWidthOverride),
      boxShadow: hardShadow(shadowColor ?? c.cardBorder, dx: dx, dy: dy),
    );
  }

  // ── Utility face — for stamped numbers: scores, dates, ids ────────────────
  static TextStyle mono(
          {double fontSize = 13,
          FontWeight weight = FontWeight.w600,
          Color? color}) =>
      GoogleFonts.ibmPlexMono(
          fontSize: fontSize, fontWeight: weight, color: color);

  // ── Dark ThemeData ─────────────────────────────────────────────────────────
  static ThemeData get dark => _build(Brightness.dark);

  // ── Light ThemeData ────────────────────────────────────────────────────────
  static ThemeData get light => _build(Brightness.light);

  static ThemeData _build(Brightness brightness) {
    final isDark = brightness == Brightness.dark;

    final Color bg = isDark ? navy900 : light50;
    final Color surface = isDark ? navy800 : Colors.white;
    final Color border = isDark ? cardBorder : ltCardBorder;
    final Color div = isDark ? divider : ltDivider;
    final Color txtPrimary = isDark ? textPrimary : ltTextPrimary;
    final Color txtSecond = isDark ? textSecondary : ltTextSecondary;
    final Color txtThird = isDark ? textTertiary : ltTextTertiary;
    final Color brand = isDark ? teal500 : const Color(0xFF149A52);

    return ThemeData(
      useMaterial3: true,
      brightness: brightness,
      scaffoldBackgroundColor: bg,
      colorScheme: ColorScheme(
        brightness: brightness,
        primary: brand,
        secondary: amber500,
        surface: surface,
        background: bg,
        error: red500,
        onPrimary: ink,
        onSecondary: ink,
        onSurface: txtPrimary,
        onBackground: txtPrimary,
        onError: Colors.white,
      ),
      extensions: [
        AppColors(
          navy900: isDark ? navy900 : light50,
          navy800: isDark ? navy800 : Colors.white,
          navy700: isDark ? navy700 : light100,
          navy600: isDark ? navy600 : light200,
          navy500: isDark ? navy500 : light300,
          navy400: isDark ? navy400 : light400,
          cardBorder: border,
          divider: div,
          textPrimary: txtPrimary,
          textSecondary: txtSecond,
          textTertiary: txtThird,
          brand: brand,
          brandSoft: isDark ? teal400 : tealDark,
        ),
      ],
      textTheme: GoogleFonts.workSansTextTheme().copyWith(
        displayLarge: GoogleFonts.spaceGrotesk(
            fontSize: 32,
            fontWeight: FontWeight.w700,
            color: txtPrimary,
            letterSpacing: -0.5),
        displayMedium: GoogleFonts.spaceGrotesk(
            fontSize: 26, fontWeight: FontWeight.w700, color: txtPrimary),
        headlineLarge: GoogleFonts.spaceGrotesk(
            fontSize: 22, fontWeight: FontWeight.w700, color: txtPrimary),
        headlineMedium: GoogleFonts.spaceGrotesk(
            fontSize: 18, fontWeight: FontWeight.w700, color: txtPrimary),
        titleLarge: GoogleFonts.spaceGrotesk(
            fontSize: 17, fontWeight: FontWeight.w700, color: txtPrimary),
        titleMedium: GoogleFonts.spaceGrotesk(
            fontSize: 14, fontWeight: FontWeight.w700, color: txtPrimary),
        bodyLarge: GoogleFonts.workSans(
            fontSize: 15,
            fontWeight: FontWeight.w400,
            color: txtPrimary,
            height: 1.6),
        bodyMedium: GoogleFonts.workSans(
            fontSize: 13,
            fontWeight: FontWeight.w500,
            color: txtSecond,
            height: 1.5),
        labelLarge: GoogleFonts.workSans(
            fontSize: 13, fontWeight: FontWeight.w700, letterSpacing: 0.3),
        labelSmall: GoogleFonts.workSans(
            fontSize: 11,
            fontWeight: FontWeight.w600,
            color: txtThird,
            letterSpacing: 0.6),
      ),
      cardTheme: CardThemeData(
        color: surface,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radius),
          side: BorderSide(color: border, width: borderWidth),
        ),
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: bg,
        elevation: 0,
        scrolledUnderElevation: 0,
        titleTextStyle: GoogleFonts.spaceGrotesk(
            fontSize: 18, fontWeight: FontWeight.w700, color: txtPrimary),
        iconTheme: IconThemeData(color: txtPrimary),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: brand,
          foregroundColor: ink,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(radiusSm),
            side: BorderSide(color: border, width: borderWidth),
          ),
          textStyle: GoogleFonts.workSans(
              fontSize: 14, fontWeight: FontWeight.w700, letterSpacing: 0.2),
        ).copyWith(
          shadowColor: const WidgetStatePropertyAll(Colors.transparent),
        ),
      ),
      // Boxed fields with a thick full border — brutalist form field.
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: isDark ? navy800 : Colors.white,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(radiusSm),
          borderSide: BorderSide(color: border, width: borderWidth),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(radiusSm),
          borderSide: BorderSide(color: border, width: borderWidth),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(radiusSm),
          borderSide: BorderSide(color: brand, width: borderWidth),
        ),
        hintStyle: GoogleFonts.workSans(color: txtThird, fontSize: 14),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
      ),
      dividerTheme: DividerThemeData(color: div, thickness: 2, space: 1),
      chipTheme: ChipThemeData(
        backgroundColor: surface,
        side: BorderSide(color: border, width: 2),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        labelStyle: GoogleFonts.workSans(
            fontSize: 12, color: txtSecond, fontWeight: FontWeight.w600),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      ),
    );
  }
}

// ── ThemeExtension so widgets can access semantic tokens without duplicating ──
class AppColors extends ThemeExtension<AppColors> {
  final Color navy900, navy800, navy700, navy600, navy500, navy400;
  final Color cardBorder, divider;
  final Color textPrimary, textSecondary, textTertiary;
  final Color brand, brandSoft;

  const AppColors({
    required this.navy900,
    required this.navy800,
    required this.navy700,
    required this.navy600,
    required this.navy500,
    required this.navy400,
    required this.cardBorder,
    required this.divider,
    required this.textPrimary,
    required this.textSecondary,
    required this.textTertiary,
    required this.brand,
    required this.brandSoft,
  });

  @override
  AppColors copyWith({
    Color? navy900,
    Color? navy800,
    Color? navy700,
    Color? navy600,
    Color? navy500,
    Color? navy400,
    Color? cardBorder,
    Color? divider,
    Color? textPrimary,
    Color? textSecondary,
    Color? textTertiary,
    Color? brand,
    Color? brandSoft,
  }) =>
      AppColors(
        navy900: navy900 ?? this.navy900,
        navy800: navy800 ?? this.navy800,
        navy700: navy700 ?? this.navy700,
        navy600: navy600 ?? this.navy600,
        navy500: navy500 ?? this.navy500,
        navy400: navy400 ?? this.navy400,
        cardBorder: cardBorder ?? this.cardBorder,
        divider: divider ?? this.divider,
        textPrimary: textPrimary ?? this.textPrimary,
        textSecondary: textSecondary ?? this.textSecondary,
        textTertiary: textTertiary ?? this.textTertiary,
        brand: brand ?? this.brand,
        brandSoft: brandSoft ?? this.brandSoft,
      );

  @override
  AppColors lerp(AppColors? other, double t) {
    if (other == null) return this;
    return AppColors(
      navy900: Color.lerp(navy900, other.navy900, t)!,
      navy800: Color.lerp(navy800, other.navy800, t)!,
      navy700: Color.lerp(navy700, other.navy700, t)!,
      navy600: Color.lerp(navy600, other.navy600, t)!,
      navy500: Color.lerp(navy500, other.navy500, t)!,
      navy400: Color.lerp(navy400, other.navy400, t)!,
      cardBorder: Color.lerp(cardBorder, other.cardBorder, t)!,
      divider: Color.lerp(divider, other.divider, t)!,
      textPrimary: Color.lerp(textPrimary, other.textPrimary, t)!,
      textSecondary: Color.lerp(textSecondary, other.textSecondary, t)!,
      textTertiary: Color.lerp(textTertiary, other.textTertiary, t)!,
      brand: Color.lerp(brand, other.brand, t)!,
      brandSoft: Color.lerp(brandSoft, other.brandSoft, t)!,
    );
  }

  static AppColors of(BuildContext context) =>
      Theme.of(context).extension<AppColors>()!;
}
