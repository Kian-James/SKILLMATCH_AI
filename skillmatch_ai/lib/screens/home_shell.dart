// lib/screens/home_shell.dart

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

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

    // For guest users just pop back to login — no Firebase session to clear.
    if (mounted) {
      Navigator.pushAndRemoveUntil(
        context,
        MaterialPageRoute(builder: (_) => const LoginScreen()),
        (_) => false,
      );
    }
  }

  Future<bool> _confirmSignOut() async {
    return await showDialog<bool>(
          context: context,
          builder: (_) => AlertDialog(
            backgroundColor: AppTheme.navy700,
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16)),
            title: Text('Sign out?',
                style: GoogleFonts.spaceGrotesk(
                    color: AppTheme.textPrimary,
                    fontWeight: FontWeight.w600)),
            content: Text(
              widget.user.isGuest
                  ? 'Your guest session and scan history will be lost.'
                  : 'You\'ll be signed out of your account.',
              style: GoogleFonts.inter(
                  color: AppTheme.textSecondary, fontSize: 13),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context, false),
                child: Text('Cancel',
                    style: GoogleFonts.inter(color: AppTheme.textSecondary)),
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
    return Scaffold(
      appBar: AppBar(
        backgroundColor: AppTheme.navy900,
        elevation: 0,
        scrolledUnderElevation: 0,
        leadingWidth: 0,
        titleSpacing: 20,
        title: Row(children: [
          // Avatar
          AvatarCircle(initials: widget.user.avatarInitials, size: 34),
          const SizedBox(width: 10),
          // Name + email/guest badge
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(widget.user.name,
                    style: GoogleFonts.spaceGrotesk(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: AppTheme.textPrimary)),
                if (widget.user.isGuest)
                  Container(
                    margin: const EdgeInsets.only(top: 2),
                    padding: const EdgeInsets.symmetric(
                        horizontal: 6, vertical: 1),
                    decoration: BoxDecoration(
                      color: AppTheme.amber500.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(4),
                      border: Border.all(
                          color: AppTheme.amber500.withOpacity(0.4),
                          width: 0.8),
                    ),
                    child: Text('Guest',
                        style: GoogleFonts.inter(
                            fontSize: 10,
                            color: AppTheme.amber400,
                            fontWeight: FontWeight.w600)),
                  )
                else if (widget.user.email.isNotEmpty)
                  Text(widget.user.email,
                      style: GoogleFonts.inter(
                          fontSize: 11, color: AppTheme.textTertiary),
                      overflow: TextOverflow.ellipsis),
              ],
            ),
          ),
        ]),
        actions: [
          // Sign out / exit button
          IconButton(
            onPressed: _signOut,
            tooltip: widget.user.isGuest ? 'Exit guest session' : 'Sign out',
            icon: Icon(
              widget.user.isGuest
                  ? Icons.exit_to_app_rounded
                  : Icons.logout_rounded,
              color: AppTheme.textTertiary,
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
        decoration: const BoxDecoration(
          color: AppTheme.navy800,
          border: Border(top: BorderSide(color: AppTheme.cardBorder)),
        ),
        child: SafeArea(
          child: Padding(
            padding:
                const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _navItem(0, Icons.upload_file_outlined,
                    Icons.upload_file_rounded, 'Analyze'),
                _navItem(1, Icons.timeline_outlined,
                    Icons.timeline_rounded, 'History'),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _navItem(
      int index, IconData icon, IconData activeIcon, String label) {
    final active = _tab == index;
    return GestureDetector(
      onTap: () => setState(() => _tab = index),
      behavior: HitTestBehavior.opaque,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding:
            const EdgeInsets.symmetric(horizontal: 24, vertical: 10),
        decoration: BoxDecoration(
          color: active
              ? AppTheme.teal500.withOpacity(0.12)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(children: [
          Icon(active ? activeIcon : icon,
              color: active ? AppTheme.teal500 : AppTheme.textTertiary,
              size: 20),
          if (active) ...[
            const SizedBox(width: 7),
            Text(label,
                style: GoogleFonts.spaceGrotesk(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.teal500)),
          ],
        ]),
      ),
    );
  }
}
