// lib/screens/login_screen.dart
// Email/password · Google Sign-In · Guest mode · Sign-up · Forgot password

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/models.dart';
import '../services/auth_service.dart';
import '../theme.dart';
import 'home_shell.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen>
    with SingleTickerProviderStateMixin {
  final _auth = AuthService();
  final _emailCtrl = TextEditingController();
  final _passCtrl = TextEditingController();
  final _nameCtrl = TextEditingController();
  final _formKey = GlobalKey<FormState>();

  bool _isSignUp = false;
  bool _obscurePass = true;
  bool _isLoading = false;
  String? _error;

  late AnimationController _fadeCtrl;
  late Animation<double> _fadeAnim;

  @override
  void initState() {
    super.initState();
    _fadeCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 350));
    _fadeAnim = CurvedAnimation(parent: _fadeCtrl, curve: Curves.easeOut);
    _fadeCtrl.forward();
  }

  @override
  void dispose() {
    _fadeCtrl.dispose();
    _emailCtrl.dispose();
    _passCtrl.dispose();
    _nameCtrl.dispose();
    super.dispose();
  }

  // ── Navigation ────────────────────────────────────────────────────────────

  void _goHome(ResumeUser user) {
    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => HomeShell(user: user)),
    );
  }

  // ── Auth actions ──────────────────────────────────────────────────────────

  Future<void> _submitEmailPassword() async {
    if (!_formKey.currentState!.validate()) return;
    _setLoading(true);
    try {
      final user = _isSignUp
          ? await _auth.signUpWithEmail(
              name: _nameCtrl.text.trim(),
              email: _emailCtrl.text.trim(),
              password: _passCtrl.text,
            )
          : await _auth.signInWithEmail(
              email: _emailCtrl.text.trim(),
              password: _passCtrl.text,
            );
      _goHome(user);
    } on FirebaseAuthException catch (e) {
      _setError(_friendlyFirebaseError(e.code));
    } catch (e) {
      _setError(e.toString().replaceFirst('Exception: ', ''));
    } finally {
      _setLoading(false);
    }
  }

  Future<void> _signInGoogle() async {
    _setLoading(true);
    try {
      final user = await _auth.signInWithGoogle();
      _goHome(user);
    } on FirebaseAuthException catch (e) {
      _setError(_friendlyFirebaseError(e.code));
    } catch (e) {
      final msg = e.toString();
      if (!msg.contains('cancelled'))
        _setError(msg.replaceFirst('Exception: ', ''));
    } finally {
      _setLoading(false);
    }
  }

  Future<void> _continueAsGuest() async {
    _setLoading(true);
    try {
      final user = await _auth.continueAsGuest();
      _goHome(user);
    } on FirebaseAuthException catch (e) {
      _setError(_friendlyFirebaseError(e.code));
    } catch (e) {
      _setError(e.toString().replaceFirst('Exception: ', ''));
    } finally {
      _setLoading(false);
    }
  }

  Future<void> _forgotPassword() async {
    final email = _emailCtrl.text.trim();
    if (email.isEmpty) {
      _setError('Enter your email above first.');
      return;
    }
    try {
      await _auth.sendPasswordReset(email);
      if (mounted) {
        _setError(null);
        _showSnack('Password reset email sent to $email');
      }
    } on FirebaseAuthException catch (e) {
      _setError(_friendlyFirebaseError(e.code));
    }
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  void _setLoading(bool v) {
    if (mounted) setState(() => _isLoading = v);
  }

  void _setError(String? v) {
    if (mounted) setState(() => _error = v);
  }

  void _toggleMode() {
    setState(() {
      _isSignUp = !_isSignUp;
      _error = null;
    });
    _fadeCtrl.forward(from: 0);
  }

  String _friendlyFirebaseError(String code) => switch (code) {
        'user-not-found' => 'No account found with that email.',
        'wrong-password' => 'Incorrect password.',
        'invalid-credential' => 'Invalid email or password.',
        'email-already-in-use' => 'An account already exists with this email.',
        'weak-password' => 'Password must be at least 6 characters.',
        'invalid-email' => 'Please enter a valid email address.',
        'too-many-requests' => 'Too many attempts. Try again later.',
        'network-request-failed' => 'No internet connection.',
        _ => 'Authentication failed. Please try again.',
      };

  void _showSnack(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(msg,
          style: GoogleFonts.workSans(color: AppColors.of(context).navy900)),
      backgroundColor: AppTheme.teal500,
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
    ));
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: FadeTransition(
          opacity: _fadeAnim,
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 24),
            child: Form(
              key: _formKey,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 24),
                  _buildHero(),
                  const SizedBox(height: 40),
                  _buildFormFields(),
                  const SizedBox(height: 8),
                  if (!_isSignUp) _buildForgotPassword(),
                  const SizedBox(height: 20),
                  if (_error != null) _buildError(),
                  _buildSubmitButton(),
                  const SizedBox(height: 20),
                  _buildDivider(),
                  const SizedBox(height: 20),
                  _buildGoogleButton(),
                  const SizedBox(height: 14),
                  _buildGuestButton(),
                  const SizedBox(height: 28),
                  _buildToggleMode(),
                  const SizedBox(height: 24),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  // ── Widgets ───────────────────────────────────────────────────────────────

  Widget _buildHero() => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 56,
            height: 56,
            decoration: BoxDecoration(
              color: AppColors.of(context).brand,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                  color: AppColors.of(context).cardBorder,
                  width: AppTheme.borderWidth),
              boxShadow: AppTheme.hardShadow(AppColors.of(context).cardBorder,
                  dx: 4, dy: 4),
            ),
            child: const Icon(Icons.auto_awesome_rounded,
                color: AppTheme.ink, size: 26),
          ),
          const SizedBox(height: 20),
          Text(
            _isSignUp ? 'Create account' : 'Welcome back',
            style: GoogleFonts.spaceGrotesk(
              fontSize: 32,
              fontWeight: FontWeight.w700,
              color: AppColors.of(context).textPrimary,
              height: 1.1,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            _isSignUp
                ? 'Start tracking your skill growth today.'
                : 'Sign in to access your resume history.',
            style: GoogleFonts.workSans(
                fontSize: 15,
                color: AppColors.of(context).textSecondary,
                height: 1.5),
          ),
        ],
      );

  Widget _buildFormFields() => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Name field — sign-up only
          AnimatedSize(
            duration: const Duration(milliseconds: 250),
            curve: Curves.easeInOut,
            child: _isSignUp
                ? Column(children: [
                    _label('Full Name'),
                    const SizedBox(height: 8),
                    TextFormField(
                      controller: _nameCtrl,
                      enabled: !_isLoading,
                      textCapitalization: TextCapitalization.words,
                      style: _inputStyle,
                      decoration: _inputDeco(
                        hint: 'Kian Reyes',
                        icon: Icons.person_outline_rounded,
                      ),
                      validator: (v) =>
                          _isSignUp && (v == null || v.trim().isEmpty)
                              ? 'Enter your name'
                              : null,
                    ),
                    const SizedBox(height: 16),
                  ])
                : const SizedBox.shrink(),
          ),

          _label('Email'),
          const SizedBox(height: 8),
          TextFormField(
            controller: _emailCtrl,
            enabled: !_isLoading,
            keyboardType: TextInputType.emailAddress,
            autocorrect: false,
            style: _inputStyle,
            decoration: _inputDeco(
              hint: 'you@gmail.com',
              icon: Icons.email_outlined,
            ),
            validator: (v) {
              if (v == null || v.trim().isEmpty) return 'Enter your email';
              if (!v.contains('@') || !v.contains('.'))
                return 'Enter a valid email';
              return null;
            },
          ),
          const SizedBox(height: 16),

          _label('Password'),
          const SizedBox(height: 8),
          TextFormField(
            controller: _passCtrl,
            enabled: !_isLoading,
            obscureText: _obscurePass,
            style: _inputStyle,
            decoration: _inputDeco(
              hint: '••••••••',
              icon: Icons.lock_outline_rounded,
            ).copyWith(
              suffixIcon: GestureDetector(
                onTap: () => setState(() => _obscurePass = !_obscurePass),
                child: Icon(
                  _obscurePass
                      ? Icons.visibility_outlined
                      : Icons.visibility_off_outlined,
                  color: AppColors.of(context).textTertiary,
                  size: 18,
                ),
              ),
            ),
            validator: (v) {
              if (v == null || v.isEmpty) return 'Enter your password';
              if (_isSignUp && v.length < 6) return 'Minimum 6 characters';
              return null;
            },
            onFieldSubmitted: (_) => _submitEmailPassword(),
          ),
        ],
      );

  Widget _buildForgotPassword() => Align(
        alignment: Alignment.centerRight,
        child: GestureDetector(
          onTap: _forgotPassword,
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Text(
              'Forgot password?',
              style: GoogleFonts.workSans(
                fontSize: 13,
                color: AppTheme.teal400,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ),
      );

  Widget _buildError() => Container(
        margin: const EdgeInsets.only(bottom: 16),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: AppTheme.red500.withOpacity(0.08),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppTheme.red500.withOpacity(0.3), width: 2.5),
        ),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Icon(Icons.error_outline_rounded,
              color: AppTheme.red400, size: 16),
          const SizedBox(width: 10),
          Expanded(
              child: Text(_error!,
                  style: GoogleFonts.workSans(
                      fontSize: 13, color: AppTheme.red400))),
        ]),
      );

  Widget _buildSubmitButton() => SizedBox(
        width: double.infinity,
        child: ElevatedButton(
          onPressed: _isLoading ? null : _submitEmailPassword,
          style: ElevatedButton.styleFrom(
            backgroundColor: AppTheme.teal500,
            foregroundColor: AppColors.of(context).navy900,
            padding: const EdgeInsets.symmetric(vertical: 15),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
              side: BorderSide(
                  color: AppColors.of(context).cardBorder,
                  width: AppTheme.borderWidth),
            ),
            disabledBackgroundColor: AppTheme.teal500.withOpacity(0.4),
          ),
          child: _isLoading
              ? SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(
                      strokeWidth: 2, color: AppColors.of(context).navy900),
                )
              : Text(
                  _isSignUp ? 'Create Account' : 'Sign In',
                  style: GoogleFonts.spaceGrotesk(
                      fontSize: 15, fontWeight: FontWeight.w700),
                ),
        ),
      );

  Widget _buildDivider() => Row(children: [
        Expanded(
            child: Container(height: 1, color: AppColors.of(context).divider)),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 14),
          child: Text('or continue with',
              style: GoogleFonts.workSans(
                  fontSize: 12,
                  color: AppColors.of(context).textTertiary,
                  letterSpacing: 0.3)),
        ),
        Expanded(
            child: Container(height: 1, color: AppColors.of(context).divider)),
      ]);

  Widget _buildGoogleButton() => SizedBox(
        width: double.infinity,
        child: OutlinedButton(
          onPressed: _isLoading ? null : _signInGoogle,
          style: OutlinedButton.styleFrom(
            foregroundColor: AppColors.of(context).textPrimary,
            side: BorderSide(
                color: AppColors.of(context).cardBorder,
                width: AppTheme.borderWidth),
            backgroundColor: AppColors.of(context).navy800,
            padding: const EdgeInsets.symmetric(vertical: 14),
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
          child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            // Google G logo using coloured squares
            _googleIcon(),
            const SizedBox(width: 12),
            Text('Continue with Google',
                style: GoogleFonts.spaceGrotesk(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: AppColors.of(context).textPrimary)),
          ]),
        ),
      );

  Widget _buildGuestButton() => SizedBox(
        width: double.infinity,
        child: OutlinedButton.icon(
          onPressed: _isLoading ? null : _continueAsGuest,
          style: OutlinedButton.styleFrom(
            foregroundColor: AppColors.of(context).textSecondary,
            side: BorderSide(
                color: AppColors.of(context).cardBorder,
                width: AppTheme.borderWidth),
            backgroundColor: Colors.transparent,
            padding: const EdgeInsets.symmetric(vertical: 14),
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
          icon: Icon(Icons.person_outline_rounded,
              color: AppColors.of(context).textTertiary, size: 18),
          label: Text('Continue as Guest',
              style: GoogleFonts.spaceGrotesk(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: AppColors.of(context).textSecondary)),
        ),
      );

  Widget _buildToggleMode() => Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            _isSignUp ? 'Already have an account? ' : "Don't have an account? ",
            style: GoogleFonts.workSans(
                fontSize: 14, color: AppColors.of(context).textTertiary),
          ),
          GestureDetector(
            onTap: _toggleMode,
            child: Text(
              _isSignUp ? 'Sign In' : 'Sign Up',
              style: GoogleFonts.spaceGrotesk(
                fontSize: 14,
                fontWeight: FontWeight.w700,
                color: AppTheme.teal400,
              ),
            ),
          ),
        ],
      );

  // ── Google icon (no asset needed) ─────────────────────────────────────────

  Widget _googleIcon() => SizedBox(
        width: 20,
        height: 20,
        child: CustomPaint(
            painter:
                _GoogleIconPainter(background: AppColors.of(context).navy800)),
      );

  // ── Style helpers ─────────────────────────────────────────────────────────

  TextStyle get _inputStyle => GoogleFonts.workSans(
      fontSize: 14, color: AppColors.of(context).textPrimary);

  InputDecoration _inputDeco({required String hint, required IconData icon}) =>
      InputDecoration(
        hintText: hint,
        prefixIcon:
            Icon(icon, color: AppColors.of(context).textTertiary, size: 18),
      );

  Widget _label(String text) => Text(text,
      style: GoogleFonts.spaceGrotesk(
          fontSize: 13,
          fontWeight: FontWeight.w600,
          color: AppColors.of(context).textSecondary));
}

// ── Google coloured G icon painter ───────────────────────────────────────────

class _GoogleIconPainter extends CustomPainter {
  final Color background;
  const _GoogleIconPainter({required this.background});

  @override
  void paint(Canvas canvas, Size size) {
    final c = Offset(size.width / 2, size.height / 2);
    final r = size.width / 2;

    // Draw four coloured quadrants
    final colors = [
      const Color(0xFF4285F4), // blue  — top-right
      const Color(0xFF34A853), // green — bottom-right
      const Color(0xFFFBBC05), // yellow — bottom-left
      const Color(0xFFEA4335), // red   — top-left
    ];

    for (int i = 0; i < 4; i++) {
      final paint = Paint()..color = colors[i];
      canvas.drawArc(
        Rect.fromCircle(center: c, radius: r),
        (-90 + i * 90) * 3.14159 / 180,
        90 * 3.14159 / 180,
        true,
        paint,
      );
    }

    // Background-coloured center circle to make it look like the G
    canvas.drawCircle(c, r * 0.55, Paint()..color = background);

    // Background-coloured horizontal bar of the G
    canvas.drawRect(
      Rect.fromLTWH(c.dx - 0.5, c.dy - r * 0.15, r * 0.95, r * 0.3),
      Paint()..color = background,
    );
  }

  @override
  bool shouldRepaint(covariant _GoogleIconPainter oldDelegate) =>
      oldDelegate.background != background;
}
