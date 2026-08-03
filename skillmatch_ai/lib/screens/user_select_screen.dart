// lib/screens/user_select_screen.dart
// Simple user name entry — enter your name, get your personalized dashboard

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/models.dart';
import '../theme.dart';
import 'home_shell.dart';

// Quick demo users — tap to go straight in
const _demoUsers = ['Kian', 'Alex', 'Maria'];

class UserSelectScreen extends StatefulWidget {
  const UserSelectScreen({super.key});

  @override
  State<UserSelectScreen> createState() => _UserSelectScreenState();
}

class _UserSelectScreenState extends State<UserSelectScreen> {
  final _ctrl = TextEditingController();
  final _form = GlobalKey<FormState>();

  void _enter(String name) {
    if (name.trim().isEmpty) return;
    final user = ResumeUser.fromName(name.trim());
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => HomeShell(user: user)),
    );
  }

  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 32),
          child: Form(
            key: _form,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Spacer(flex: 2),

                // Logo / hero
                Container(
                  width: 60,
                  height: 60,
                  decoration: BoxDecoration(
                    color: AppTheme.teal500,
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: AppTheme.ink, width: AppTheme.borderWidth),
                    boxShadow: AppTheme.hardShadow(AppTheme.ink, dx: 4, dy: 4),
                  ),
                  child: const Icon(Icons.auto_awesome_rounded,
                      color: AppTheme.ink, size: 28),
                ),
                const SizedBox(height: 24),

                Text('Resume\nAnalyzer',
                    style: GoogleFonts.spaceGrotesk(
                      fontSize: 38,
                      fontWeight: FontWeight.w700,
                      color: AppTheme.textPrimary,
                      height: 1.1,
                    )),
                const SizedBox(height: 10),
                Text('AI-powered skill gap analysis.\nTrack your growth across every scan.',
                    style: GoogleFonts.inter(
                        fontSize: 15,
                        color: AppTheme.textSecondary,
                        height: 1.6)),

                const Spacer(flex: 2),

                // Name field
                Text('Your name',
                    style: GoogleFonts.spaceGrotesk(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: AppTheme.textSecondary)),
                const SizedBox(height: 8),
                TextFormField(
                  controller: _ctrl,
                  autofocus: false,
                  style: GoogleFonts.inter(fontSize: 15, color: AppTheme.textPrimary),
                  decoration: const InputDecoration(
                    hintText: 'e.g. Kian',
                    prefixIcon: Icon(Icons.person_outline_rounded,
                        color: AppTheme.textTertiary, size: 18),
                  ),
                  onFieldSubmitted: _enter,
                  validator: (v) => (v == null || v.trim().isEmpty)
                      ? 'Enter your name to continue'
                      : null,
                ),
                const SizedBox(height: 16),

                // Demo user quick-select
                Wrap(spacing: 8, children: _demoUsers.map((n) => GestureDetector(
                  onTap: () => _enter(n),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                    decoration: BoxDecoration(
                      color: AppTheme.navy700,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: AppTheme.cardBorder, width: 2.5),
                    ),
                    child: Row(mainAxisSize: MainAxisSize.min, children: [
                      Container(
                        width: 22, height: 22,
                        decoration: BoxDecoration(
                          color: AppTheme.teal500.withOpacity(0.15),
                          shape: BoxShape.circle,
                        ),
                        child: Center(child: Text(n[0],
                            style: GoogleFonts.spaceGrotesk(
                                fontSize: 11, color: AppTheme.teal400,
                                fontWeight: FontWeight.w700))),
                      ),
                      const SizedBox(width: 7),
                      Text(n, style: GoogleFonts.inter(
                          fontSize: 13, color: AppTheme.textSecondary,
                          fontWeight: FontWeight.w500)),
                    ]),
                  ),
                )).toList()),

                const SizedBox(height: 24),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: () {
                      if (_form.currentState!.validate()) _enter(_ctrl.text);
                    },
                    icon: const Icon(Icons.arrow_forward_rounded, size: 18),
                    label: const Text('Get Started'),
                  ),
                ),

                const Spacer(flex: 3),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
