// lib/screens/home_shell.dart

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../main.dart';
import '../models/models.dart';
import '../services/auth_service.dart';
import '../theme.dart';
import '../widgets/common_widgets.dart';
import 'history_screen.dart';
import 'login_screen.dart';
import 'upload_screen.dart';

class HomeShell extends StatefulWidget {
  final ResumeUser user;
  const HomeShell({super.key, required this.user});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _tab = 0;
  final _auth = AuthService();

  Future<void> _signOut() async {
    final confirmed = await _confirmSignOut();
    if (!confirmed || !mounted) return;

    if (!widget.user.isGuest) await _auth.signOut();

    if (mounted) {
      Navigator.pushAndRemoveUntil(
        context,
        MaterialPageRoute(builder: (_) => const LoginScreen()),
        (_) => false,
      );
    }
  }

  Future<bool> _confirmSignOut() async {
    final c = AppColors.of(context);
    return await showDialog<bool>(
          context: context,
          builder: (_) => AlertDialog(
            backgroundColor: c.navy700,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            title: Text('Sign out?',
                style: GoogleFonts.spaceGrotesk(
                    color: c.textPrimary, fontWeight: FontWeight.w600)),
            content: Text(
              widget.user.isGuest
                  ? 'Your guest session and scan history will be lost.'
                  : 'You\'ll be signed out of your account.',
              style: GoogleFonts.workSans(color: c.textSecondary, fontSize: 13),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context, false),
                child: Text('Cancel',
                    style: GoogleFonts.workSans(color: c.textSecondary)),
              ),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.red500,
                  foregroundColor: Colors.white,
                ),
                onPressed: () => Navigator.pop(context, true),
                child: Text(widget.user.isGuest ? 'Exit' : 'Sign Out'),
              ),
            ],
          ),
        ) ??
        false;
  }

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    final appState = ResumeAnalyzerApp.of(context);
    final isDark = appState.isDark;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: c.navy900,
        elevation: 0,
        scrolledUnderElevation: 0,
        leadingWidth: 0,
        titleSpacing: 20,
        title: Row(children: [
          AvatarCircle(initials: widget.user.avatarInitials, size: 34),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(widget.user.name,
                    style: GoogleFonts.spaceGrotesk(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: c.textPrimary)),
                if (widget.user.isGuest)
                  Container(
                    margin: const EdgeInsets.only(top: 2),
                    padding:
                        const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                    decoration: BoxDecoration(
                      color: AppTheme.amber500.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(4),
                      border: Border.all(
                          color: AppTheme.amber500.withOpacity(0.4),
                          width: 0.8),
                    ),
                    child: Text('Guest',
                        style: GoogleFonts.workSans(
                            fontSize: 10,
                            color: AppTheme.amber400,
                            fontWeight: FontWeight.w600)),
                  )
                else if (widget.user.email.isNotEmpty)
                  Text(widget.user.email,
                      style: GoogleFonts.workSans(
                          fontSize: 11, color: c.textTertiary),
                      overflow: TextOverflow.ellipsis),
              ],
            ),
          ),
        ]),
        actions: [
          // ── Light / dark toggle ─────────────────────────────────────
          IconButton(
            onPressed: appState.toggleTheme,
            tooltip: isDark ? 'Switch to light mode' : 'Switch to dark mode',
            icon: AnimatedSwitcher(
              duration: const Duration(milliseconds: 300),
              transitionBuilder: (child, anim) =>
                  RotationTransition(turns: anim, child: child),
              child: Icon(
                isDark ? Icons.wb_sunny_outlined : Icons.nightlight_round,
                key: ValueKey(isDark),
                color: c.textTertiary,
                size: 20,
              ),
            ),
          ),
          // ── Sign-out ────────────────────────────────────────────────
          IconButton(
            onPressed: _signOut,
            tooltip: widget.user.isGuest ? 'Exit guest session' : 'Sign out',
            icon: Icon(
              widget.user.isGuest
                  ? Icons.exit_to_app_rounded
                  : Icons.logout_rounded,
              color: c.textTertiary,
              size: 20,
            ),
          ),
          const SizedBox(width: 4),
        ],
      ),
      body: IndexedStack(
        index: _tab,
        children: [
          UploadScreen(user: widget.user),
          HistoryScreen(user: widget.user),
        ],
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: c.navy800,
          border: Border(top: BorderSide(color: c.cardBorder, width: 2.5)),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _navItem(context, 0, Icons.upload_file_outlined,
                    Icons.upload_file_rounded, 'Analyze'),
                _navItem(context, 1, Icons.timeline_outlined,
                    Icons.timeline_rounded, 'History'),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _navItem(BuildContext context, int index, IconData icon,
      IconData activeIcon, String label) {
    final active = _tab == index;
    final c = AppColors.of(context);
    return GestureDetector(
      onTap: () => setState(() => _tab = index),
      behavior: HitTestBehavior.opaque,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 8),
        decoration: BoxDecoration(
          border: Border(
            top: BorderSide(
                color: active ? c.brand : Colors.transparent, width: 2),
          ),
        ),
        child: Row(children: [
          Icon(active ? activeIcon : icon,
              color: active ? c.brand : c.textTertiary, size: 20),
          if (active) ...[
            const SizedBox(width: 7),
            Text(label,
                style: GoogleFonts.spaceGrotesk(
                    fontSize: 13, fontWeight: FontWeight.w600, color: c.brand)),
          ],
        ]),
      ),
    );
  }
}
