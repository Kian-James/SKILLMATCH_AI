// lib/theme.dart
// Direction B — "Dossier": resumes reviewed like case files.
// Light = paper case file. Dark = ink-and-lamplight reading room.

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  // ── Dark palette ("night dossier") ─────────────────────────────────────────
  static const Color navy900 = Color(0xFF11150F);
  static const Color navy800 = Color(0xFF1A1F19);
  static const Color navy700 = Color(0xFF232A21);
  static const Color navy600 = Color(0xFF2C332A);
  static const Color navy500 = Color(0xFF3A4438);
  static const Color navy400 = Color(0xFF4C5949);

  // ── Light palette (paper) ────────────────────────────────────────────────
  static const Color light50 = Color(0xFFF1F4F1);
  static const Color light100 = Color(0xFFEAEDE9);
  static const Color light200 = Color(0xFFDDE3DC);
  static const Color light300 = Color(0xFFC7D0C7);
  static const Color light400 = Color(0xFF93A199);

  // ── Brand — forest (accent) ──────────────────────────────────────────────
  static const Color teal500 =
      Color(0xFF4C9B72); // bright forest, dark-mode brand
  static const Color teal400 = Color(0xFF6BB088);
  static const Color teal300 = Color(0xFF8FC7A5);
  static const Color teal200 = Color(0xFFB9DFC7);
  static const Color teal100 = Color(0xFFE0F0E6);

  // Forest shifted darker for light-mode text (meets contrast ratio)
  static const Color tealDark = Color(0xFF1F5C4C);

  // Gold — credential / warning accent
  static const Color amber500 = Color(0xFFA9782F);
  static const Color amber400 = Color(0xFFC79A52);
  // Brick — danger
  static const Color red500 = Color(0xFFA6402E);
  static const Color red400 = Color(0xFFC1614C);
  // Sage — success
  static const Color green500 = Color(0xFF3E7A55);
  static const Color green400 = Color(0xFF5C9873);

  // ── Dark semantic tokens ───────────────────────────────────────────────────
  static const Color textPrimary = Color(0xFFEDEFE8);
  static const Color textSecondary = Color(0xFFAAB3A3);
  static const Color textTertiary = Color(0xFF6E796A);
  static const Color cardBorder = Color(0xFF2C332A);
  static const Color divider = Color(0xFF232A21);

  // ── Light semantic tokens ─────────────────────────────────────────────────
  static const Color ltTextPrimary = Color(0xFF1C2B26);
  static const Color ltTextSecondary = Color(0xFF5C6D64);
  static const Color ltTextTertiary = Color(0xFF93A199);
  static const Color ltCardBorder = Color(0xFFDDE3DC);
  static const Color ltDivider = Color(0xFFE7EAE5);
  static const Color ltCardBg = Color(0xFFFFFFFF);
  static const Color ltSurface = Color(0xFFF1F4F1);

  // ── Utility face — for stamped numbers: scores, dates, ids ────────────────
  static TextStyle mono(
          {double fontSize = 13,
          FontWeight weight = FontWeight.w500,
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
    final Color surface2 = isDark ? navy700 : light100;
    final Color border = isDark ? cardBorder : ltCardBorder;
    final Color div = isDark ? divider : ltDivider;
    final Color txtPrimary = isDark ? textPrimary : ltTextPrimary;
    final Color txtSecond = isDark ? textSecondary : ltTextSecondary;
    final Color txtThird = isDark ? textTertiary : ltTextTertiary;
    final Color brand = isDark ? teal500 : tealDark;

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
        onPrimary: Colors.white,
        onSecondary: Colors.white,
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
        displayLarge: GoogleFonts.fraunces(
            fontSize: 32,
            fontWeight: FontWeight.w600,
            color: txtPrimary,
            letterSpacing: -0.3),
        displayMedium: GoogleFonts.fraunces(
            fontSize: 26, fontWeight: FontWeight.w600, color: txtPrimary),
        headlineLarge: GoogleFonts.fraunces(
            fontSize: 22, fontWeight: FontWeight.w600, color: txtPrimary),
        headlineMedium: GoogleFonts.fraunces(
            fontSize: 18, fontWeight: FontWeight.w600, color: txtPrimary),
        titleLarge: GoogleFonts.fraunces(
            fontSize: 17, fontWeight: FontWeight.w600, color: txtPrimary),
        titleMedium: GoogleFonts.fraunces(
            fontSize: 14, fontWeight: FontWeight.w600, color: txtPrimary),
        bodyLarge: GoogleFonts.workSans(
            fontSize: 15,
            fontWeight: FontWeight.w400,
            color: txtPrimary,
            height: 1.6),
        bodyMedium: GoogleFonts.workSans(
            fontSize: 13,
            fontWeight: FontWeight.w400,
            color: txtSecond,
            height: 1.5),
        labelLarge: GoogleFonts.workSans(
            fontSize: 13, fontWeight: FontWeight.w600, letterSpacing: 0.3),
        labelSmall: GoogleFonts.workSans(
            fontSize: 11,
            fontWeight: FontWeight.w500,
            color: txtThird,
            letterSpacing: 0.6),
      ),
      cardTheme: CardThemeData(
        color: surface,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(14),
          side: BorderSide(color: border, width: 1),
        ),
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: bg,
        elevation: 0,
        scrolledUnderElevation: 0,
        titleTextStyle: GoogleFonts.fraunces(
            fontSize: 18, fontWeight: FontWeight.w600, color: txtPrimary),
        iconTheme: IconThemeData(color: txtSecond),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: brand,
          foregroundColor: Colors.white,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          textStyle: GoogleFonts.workSans(
              fontSize: 14, fontWeight: FontWeight.w600, letterSpacing: 0.2),
        ),
      ),
      // Underline fields — a form on paper, not a filled app-style box.
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.transparent,
        border: UnderlineInputBorder(borderSide: BorderSide(color: border)),
        enabledBorder:
            UnderlineInputBorder(borderSide: BorderSide(color: border)),
        focusedBorder: UnderlineInputBorder(
            borderSide: BorderSide(color: brand, width: 1.5)),
        hintStyle: GoogleFonts.workSans(color: txtThird, fontSize: 14),
        contentPadding: const EdgeInsets.symmetric(horizontal: 4, vertical: 12),
      ),
      dividerTheme: DividerThemeData(color: div, thickness: 1, space: 1),
      chipTheme: ChipThemeData(
        backgroundColor: surface,
        side: BorderSide(color: border),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(7)),
        labelStyle: GoogleFonts.workSans(fontSize: 12, color: txtSecond),
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
