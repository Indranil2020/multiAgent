"""
Web Development Prompts Module.

This module provides comprehensive prompt templates for web development tasks
in the zero-error system. Covers frontend, backend, APIs, authentication,
deployment, and web-specific best practices.

Key Areas:
- Frontend development (React, Vue, vanilla JS)
- Backend API development (REST, GraphQL)
- Authentication and authorization
- Web security (XSS, CSRF, injection)
- Performance optimization (caching, CDN)
- Responsive design and accessibility
- Database integration
- Deployment and DevOps

All prompts enforce zero-error philosophy with production-ready code.
"""

from typing import Optional, List
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_prompts import PromptTemplate, PromptFormat


# React Component Implementation Prompt
REACT_COMPONENT_PROMPT = PromptTemplate(
    template_id="react_component",
    name="React Component Implementation Prompt",
    template_text="""Implement a React component with modern best practices.

COMPONENT NAME: {component_name}
DESCRIPTION: {description}
PROPS: {props}
STATE REQUIREMENTS: {state_requirements}
FEATURES: {features}

REQUIREMENTS:
1. Functional component with hooks
2. TypeScript with proper prop types
3. Proper state management (useState, useReducer)
4. Side effects with useEffect
5. Memoization where appropriate (useMemo, useCallback)
6. Accessibility (ARIA attributes, keyboard navigation)
7. Responsive design
8. Error boundaries for error handling
9. Loading and error states
10. Comprehensive JSDoc comments

COMPONENT STRUCTURE:
```typescript
import React, {{ useState, useEffect, useMemo, useCallback }} from 'react';

interface {ComponentName}Props {{
    /** Prop description */
    propName: string;
    /** Optional prop description */
    optionalProp?: number;
    /** Callback prop description */
    onAction?: (value: string) => void;
}}

/**
 * Component description.
 *
 * @example
 * ```tsx
 * <{ComponentName}
 *   propName="value"
 *   onAction={{(val) => console.log(val)}}
 * />
 * ```
 */
export const {ComponentName}: React.FC<{ComponentName}Props> = ({{
    propName,
    optionalProp = 0,
    onAction
}}) => {{
    // State management
    const [data, setData] = useState<DataType | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    // Side effects
    useEffect(() => {{
        // Fetch data or perform side effects
        const fetchData = async () => {
            setLoading(true);
            // We assume api.fetchData returns a result object or tuple (data, error)
            // instead of throwing exceptions, adhering to zero-error principles.
            const [result, err] = await api.fetchData(propName);
            
            if (err) {
                setError(err);
                setData(null);
            } else {
                setData(result);
                setError(null);
            }
            setLoading(false);
        };

        fetchData();

        // Cleanup
        return () => {{
            // Cancel requests, clear timers, etc.
        }};
    }}, [propName]);

    // Memoized values
    const processedData = useMemo(() => {{
        if (!data) return null;
        return data.map(item => processItem(item));
    }}, [data]);

    // Memoized callbacks
    const handleAction = useCallback((value: string) => {{
        onAction?.(value);
    }}, [onAction]);

    // Loading state
    if (loading) {{
        return (
            <div role="status" aria-live="polite">
                Loading...
            </div>
        );
    }}

    // Error state
    if (error) {{
        return (
            <div role="alert" aria-live="assertive">
                Error: {{error.message}}
            </div>
        );
    }}

    // Empty state
    if (!data) {{
        return <div>No data available</div>;
    }}

    // Main render
    return (
        <div className="component-name">
            {{/* Component content */}}
        </div>
    );
}};

{ComponentName}.displayName = '{ComponentName}';
```

BEST PRACTICES:
- Use semantic HTML elements
- Implement keyboard navigation
- Add ARIA labels for accessibility
- Handle loading and error states
- Avoid prop drilling (use Context if needed)
- Extract complex logic to custom hooks
- Use CSS modules or styled-components
- Implement proper event handling
- Clean up side effects

Generate complete, production-ready React component.""",
    format=PromptFormat.MARKDOWN,
    variables=["component_name", "description", "props", "state_requirements", "features"]
)


# REST API Endpoint Implementation Prompt
REST_API_PROMPT = PromptTemplate(
    template_id="rest_api_endpoint",
    name="REST API Endpoint Implementation Prompt",
    template_text="""Implement a RESTful API endpoint with best practices.

ENDPOINT: {method} {path}
DESCRIPTION: {description}
REQUEST SCHEMA: {request_schema}
RESPONSE SCHEMA: {response_schema}
AUTHENTICATION: {authentication}
AUTHORIZATION: {authorization}

REQUIREMENTS:
1. RESTful design principles
2. Proper HTTP status codes
3. Request validation with Pydantic/Marshmallow
4. Error handling with consistent format
5. Rate limiting
6. Authentication and authorization
7. Request/response logging
8. API versioning
9. CORS configuration
10. OpenAPI/Swagger documentation

FASTAPI IMPLEMENTATION:
```python
from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["{resource}"])
security = HTTPBearer()


class {Resource}CreateRequest(BaseModel):
    \"\"\"Request schema for creating {resource}.

    Attributes:
        field1: Description of field1
        field2: Description of field2
    \"\"\"
    field1: str = Field(..., min_length=1, max_length=255, description="Field1 description")
    field2: int = Field(..., gt=0, le=1000, description="Field2 description")
    optional_field: Optional[str] = Field(None, description="Optional field description")

    @validator('field1')
    def validate_field1(cls, v: str) -> str:
        \"\"\"Validate field1 format.\"\"\"
        if not v.strip():
            raise ValueError('field1 cannot be empty')
        return v.strip()

    class Config:
        schema_extra = {{
            "example": {{
                "field1": "example value",
                "field2": 42,
                "optional_field": "optional value"
            }}
        }}


class {Resource}Response(BaseModel):
    \"\"\"Response schema for {resource}.

    Attributes:
        id: Unique identifier
        field1: Field1 value
        field2: Field2 value
        created_at: Creation timestamp
        updated_at: Last update timestamp
    \"\"\"
    id: int
    field1: str
    field2: int
    optional_field: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class ErrorResponse(BaseModel):
    \"\"\"Standard error response.

    Attributes:
        error_code: Machine-readable error code
        message: Human-readable error message
        details: Additional error details
    \"\"\"
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    \"\"\"Get current authenticated user from JWT token.

    Args:
        credentials: HTTP bearer token

    Returns:
        User information

    Raises:
        HTTPException: If token is invalid or expired
    \"\"\"
    # Verify JWT token
    # Assumes verify_jwt_token returns (payload, error) tuple
    payload, error = verify_jwt_token(credentials.credentials)
    
    if error:
        logger.error(f"Authentication failed: {{error}}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
        
    return payload


async def check_permissions(
    user: Dict[str, Any],
    required_permission: str
) -> bool:
    \"\"\"Check if user has required permission.

    Args:
        user: User information
        required_permission: Required permission name

    Returns:
        True if user has permission

    Raises:
        HTTPException: If user lacks permission
    \"\"\"
    if required_permission not in user.get('permissions', []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing required permission: {{required_permission}}"
        )
    return True


@router.post(
    "/{resources}",
    response_model={Resource}Response,
    status_code=status.HTTP_201_CREATED,
    responses={{
        201: {{"description": "{Resource} created successfully"}},
        400: {{"model": ErrorResponse, "description": "Invalid request"}},
        401: {{"model": ErrorResponse, "description": "Unauthorized"}},
        403: {{"model": ErrorResponse, "description": "Forbidden"}},
        422: {{"model": ErrorResponse, "description": "Validation error"}},
        500: {{"model": ErrorResponse, "description": "Internal server error"}}
    }}
)
async def create_{resource}(
    request: {Resource}CreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    http_request: Request = None
) -> {Resource}Response:
    \"\"\"Create new {resource}.

    Args:
        request: {Resource} creation data
        current_user: Authenticated user
        http_request: HTTP request object

    Returns:
        Created {resource}

    Raises:
        HTTPException: 400 if validation fails
        HTTPException: 401 if not authenticated
        HTTPException: 403 if insufficient permissions
        HTTPException: 500 if server error
    \"\"\"
    # Log request
    logger.info(
        f"Creating {resource}",
        extra={{
            "user_id": current_user.get('user_id'),
            "ip": http_request.client.host if http_request else None,
            "data": request.dict()
        }}
    )

    # Check permissions
    await check_permissions(current_user, "create_{resource}")

    # Validate business logic
    if not validate_business_rules(request):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Business validation failed"
        )

    # Create {resource} in database
    {resource}_data = request.dict()
    {resource}_data['created_by'] = current_user.get('user_id')

    # Assumes db.create_{resource} returns (result, error)
    new_{resource}, error = await db.create_{resource}({resource}_data)

    if error:
        logger.error(f"Database error: {{error}}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create {resource}"
        )

    # Log success
    logger.info(
        f"{Resource} created successfully",
        extra={{"id": new_{resource}.id, "user_id": current_user.get('user_id')}}
    )

    return {Resource}Response.from_orm(new_{resource})


@router.get(
    "/{resources}",
    response_model=List[{Resource}Response],
    responses={{
        200: {{"description": "List of {resources}"}},
        401: {{"model": ErrorResponse}},
        500: {{"model": ErrorResponse}}
    }}
)
async def list_{resources}(
    skip: int = 0,
    limit: int = 100,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> List[{Resource}Response]:
    \"\"\"List {resources} with pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        current_user: Authenticated user

    Returns:
        List of {resources}
    \"\"\"
    # Assumes db.get_{resources} returns (results, error)
    {resources}, error = await db.get_{resources}(skip=skip, limit=limit)
    
    if error:
        logger.error(f"Error listing {resources}: {{error}}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve {resources}"
        )
        
    return [{Resource}Response.from_orm(r) for r in {resources}]
```

BEST PRACTICES:
- Use proper HTTP methods (GET, POST, PUT, PATCH, DELETE)
- Return appropriate status codes
- Validate all inputs
- Handle errors gracefully
- Implement rate limiting
- Add comprehensive logging
- Use versioning (/api/v1/)
- Document with OpenAPI
- Implement CORS properly
- Use database transactions
- Add health check endpoints
- Implement request ID tracking

Generate complete, production-ready API endpoint.""",
    format=PromptFormat.MARKDOWN,
    variables=["method", "path", "description", "request_schema", "response_schema", "authentication", "authorization"]
)


# Authentication System Implementation Prompt
AUTHENTICATION_SYSTEM_PROMPT = PromptTemplate(
    template_id="authentication_system",
    name="Authentication System Implementation Prompt",
    template_text="""Implement secure authentication system.

AUTH TYPE: {auth_type}
FEATURES: {features}
TOKEN TYPE: {token_type}
STORAGE: {storage}

REQUIREMENTS:
1. Secure password hashing (bcrypt/argon2)
2. JWT token generation and validation
3. Refresh token mechanism
4. Password reset flow
5. Email verification
6. Rate limiting on auth endpoints
7. Account lockout after failed attempts
8. Secure session management
9. CSRF protection
10. Security headers

JWT AUTHENTICATION IMPLEMENTATION:
```python
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, validator
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets
import logging

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")  # Must be from environment
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


class UserRegister(BaseModel):
    \"\"\"User registration request.

    Attributes:
        email: User email address
        password: User password (min 8 chars)
        full_name: User full name
    \"\"\"
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str = Field(..., min_length=1, max_length=255)

    @validator('password')
    def validate_password_strength(cls, v: str) -> str:
        \"\"\"Validate password meets security requirements.\"\"\"
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        if not any(c in '!@#$%^&*()_+-=[]{{}}|;:,.<>?' for c in v):
            raise ValueError('Password must contain special character')
        return v


class UserLogin(BaseModel):
    \"\"\"User login request.

    Attributes:
        email: User email address
        password: User password
    \"\"\"
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    \"\"\"Token response.

    Attributes:
        access_token: JWT access token
        refresh_token: JWT refresh token
        token_type: Token type (always "bearer")
        expires_in: Seconds until access token expires
    \"\"\"
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    \"\"\"Refresh token request.

    Attributes:
        refresh_token: JWT refresh token
    \"\"\"
    refresh_token: str


def hash_password(password: str) -> str:
    \"\"\"Hash password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password
    \"\"\"
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    \"\"\"Verify password against hash.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password

    Returns:
        True if password matches
    \"\"\"
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    \"\"\"Create JWT access token.

    Args:
        data: Payload data to encode
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    \"\"\"
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({{
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    \"\"\"Create JWT refresh token.

    Args:
        data: Payload data to encode

    Returns:
        Encoded JWT refresh token
    \"\"\"
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({{
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
        "jti": secrets.token_urlsafe(32)  # Unique token ID
    }})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    \"\"\"Verify and decode JWT token.

    Args:
        token: JWT token to verify
        token_type: Expected token type

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    \"\"\"
    # Verify and decode JWT token
    # We assume jwt.decode returns (payload, error) or we wrap it safely
    # For this template, we assume a safe wrapper exists
    
    # payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    # In a zero-error system, we would use a safe wrapper:
    payload, error = safe_jwt_decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    
    if error:
        logger.error(f"JWT verification failed: {{error}}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Verify token type
    if payload.get("type") != token_type:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token type. Expected {{token_type}}"
        )

    # Check if token is blacklisted (implement blacklist check)
    if is_token_blacklisted(payload.get("jti")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked"
        )

    return payload


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    \"\"\"Get current user from JWT token.

    Args:
        credentials: HTTP bearer credentials

    Returns:
        User information

    Raises:
        HTTPException: If authentication fails
    \"\"\"
    token_payload = verify_token(credentials.credentials, "access")
    user_id = token_payload.get("sub")

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )

    # Get user from database
    user = await db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return user


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegister) -> TokenResponse:
    \"\"\"Register new user.

    Args:
        user_data: User registration data

    Returns:
        Access and refresh tokens

    Raises:
        HTTPException: 400 if user already exists
    \"\"\"
    # Check if user exists
    existing_user = await db.get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Hash password
    hashed_password = hash_password(user_data.password)

    # Create user
    new_user = await db.create_user({{
        "email": user_data.email,
        "hashed_password": hashed_password,
        "full_name": user_data.full_name,
        "is_verified": False,
        "is_active": True,
        "login_attempts": 0
    }})

    # Send verification email
    await send_verification_email(new_user.email, new_user.id)

    # Create tokens
    access_token = create_access_token({{"sub": str(new_user.id)}})
    refresh_token = create_refresh_token({{"sub": str(new_user.id)}})

    logger.info(f"User registered: {{user_data.email}}")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/login", response_model=TokenResponse)
async def login_user(login_data: UserLogin) -> TokenResponse:
    \"\"\"Authenticate user and return tokens.

    Args:
        login_data: User login credentials

    Returns:
        Access and refresh tokens

    Raises:
        HTTPException: 401 if credentials invalid
        HTTPException: 429 if account locked
    \"\"\"
    # Get user
    user = await db.get_user_by_email(login_data.email)
    if not user:
        # Don't reveal if email exists
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Check if account is locked
    if user.locked_until and user.locked_until > datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Account locked. Try again later."
        )

    # Verify password
    if not verify_password(login_data.password, user.hashed_password):
        # Increment login attempts
        await db.increment_login_attempts(user.id)

        # Lock account after max attempts
        if user.login_attempts + 1 >= MAX_LOGIN_ATTEMPTS:
            lockout_until = datetime.utcnow() + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
            await db.lock_account(user.id, lockout_until)
            logger.warning(f"Account locked: {{login_data.email}}")

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Reset login attempts on successful login
    await db.reset_login_attempts(user.id)

    # Create tokens
    access_token = create_access_token({{"sub": str(user.id)}})
    refresh_token = create_refresh_token({{"sub": str(user.id)}})

    # Update last login
    await db.update_last_login(user.id)

    logger.info(f"User logged in: {{login_data.email}}")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(refresh_data: RefreshTokenRequest) -> TokenResponse:
    \"\"\"Refresh access token using refresh token.

    Args:
        refresh_data: Refresh token

    Returns:
        New access token

    Raises:
        HTTPException: 401 if refresh token invalid
    \"\"\"
    # Verify refresh token
    token_payload = verify_token(refresh_data.refresh_token, "refresh")
    user_id = token_payload.get("sub")

    # Create new access token
    access_token = create_access_token({{"sub": user_id}})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_data.refresh_token,  # Return same refresh token
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/logout")
async def logout_user(
    current_user: Dict[str, Any] = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, str]:
    \"\"\"Logout user by blacklisting token.

    Args:
        current_user: Current authenticated user
        credentials: Bearer token

    Returns:
        Success message
    \"\"\"
    # Blacklist token
    token_payload = verify_token(credentials.credentials, "access")
    await blacklist_token(token_payload.get("jti"), token_payload.get("exp"))

    logger.info(f"User logged out: {{current_user.get('email')}}")

    return {{"message": "Successfully logged out"}}
```

SECURITY BEST PRACTICES:
- Use strong password hashing (bcrypt, argon2)
- Implement rate limiting on auth endpoints
- Lock accounts after failed login attempts
- Use HTTPS only
- Implement CSRF protection
- Set secure cookie flags (HttpOnly, Secure, SameSite)
- Short-lived access tokens (15 minutes)
- Long-lived refresh tokens (7 days)
- Token blacklisting for logout
- Email verification
- Two-factor authentication (optional)
- Security headers (HSTS, CSP, X-Frame-Options)
- Input validation and sanitization
- Audit logging

Generate complete, production-ready authentication system.""",
    format=PromptFormat.MARKDOWN,
    variables=["auth_type", "features", "token_type", "storage"]
)


# Frontend Form Validation Prompt
FORM_VALIDATION_PROMPT = PromptTemplate(
    template_id="form_validation",
    name="Frontend Form Validation Prompt",
    template_text="""Implement frontend form with comprehensive validation.

FORM NAME: {form_name}
FIELDS: {fields}
VALIDATION RULES: {validation_rules}
SUBMISSION: {submission}

REQUIREMENTS:
1. Client-side validation (immediate feedback)
2. Server-side validation (security)
3. Accessible error messages
4. Loading states during submission
5. Success and error handling
6. Field-level and form-level validation
7. Custom validation rules
8. Debounced async validation
9. Dirty field tracking
10. Form state management

REACT HOOK FORM IMPLEMENTATION:
```typescript
import React from 'react';
import {{ useForm, SubmitHandler }} from 'react-hook-form';
import {{ zodResolver }} from '@hookform/resolvers/zod';
import * as z from 'zod';

// Validation schema with Zod
const formSchema = z.object({{
    email: z
        .string()
        .min(1, {{ message: 'Email is required' }})
        .email({{ message: 'Invalid email format' }}),

    password: z
        .string()
        .min(8, {{ message: 'Password must be at least 8 characters' }})
        .regex(/[A-Z]/, {{ message: 'Password must contain uppercase letter' }})
        .regex(/[a-z]/, {{ message: 'Password must contain lowercase letter' }})
        .regex(/[0-9]/, {{ message: 'Password must contain number' }})
        .regex(/[^A-Za-z0-9]/, {{ message: 'Password must contain special character' }}),

    confirmPassword: z.string(),

    age: z
        .number()
        .min(18, {{ message: 'Must be 18 or older' }})
        .max(120, {{ message: 'Invalid age' }}),

    terms: z
        .boolean()
        .refine((val) => val === true, {{
            message: 'You must accept the terms and conditions'
        }})
}}).refine((data) => data.password === data.confirmPassword, {{
    message: 'Passwords do not match',
    path: ['confirmPassword']
}});

type FormData = z.infer<typeof formSchema>;

interface {FormName}Props {{
    onSubmit: (data: FormData) => Promise<void>;
    initialData?: Partial<FormData>;
}}

export const {FormName}: React.FC<{FormName}Props> = ({{
    onSubmit,
    initialData
}}) => {{
    const {{
        register,
        handleSubmit,
        formState: {{ errors, isSubmitting, isDirty, isValid }},
        setError,
        reset,
        watch
    }} = useForm<FormData>({{
        resolver: zodResolver(formSchema),
        defaultValues: initialData,
        mode: 'onBlur'  // Validate on blur
    }});

    const [submitError, setSubmitError] = React.useState<string | null>(null);
    const [submitSuccess, setSubmitSuccess] = React.useState(false);

    const onSubmitHandler: SubmitHandler<FormData> = async (data) => {{
        setSubmitError(null);
        setSubmitSuccess(false);

        try {{
            await onSubmit(data);
            setSubmitSuccess(true);
            reset();  // Reset form on success
        }} catch (error) {{
            if (error.response?.data?.errors) {{
                // Set field-specific errors from server
                Object.entries(error.response.data.errors).forEach(([field, message]) => {{
                    setError(field as keyof FormData, {{
                        type: 'server',
                        message: message as string
                    }});
                }});
            }} else {{
                // Set general form error
                setSubmitError(error.message || 'An error occurred. Please try again.');
            }}
        }}
    }};

    return (
        <form
            onSubmit={{handleSubmit(onSubmitHandler)}}
            className="form"
            noValidate  // Use custom validation instead of browser validation
        >
            {{/* Email Field */}}
            <div className="form-field">
                <label htmlFor="email" className="form-label">
                    Email <span aria-label="required">*</span>
                </label>
                <input
                    id="email"
                    type="email"
                    className={{`form-input ${{errors.email ? 'error' : ''}}`}}
                    {{...register('email')}}
                    aria-invalid={{errors.email ? 'true' : 'false'}}
                    aria-describedby={{errors.email ? 'email-error' : undefined}}
                />
                {{errors.email && (
                    <p id="email-error" className="form-error" role="alert">
                        {{errors.email.message}}
                    </p>
                )}}
            </div>

            {{/* Password Field */}}
            <div className="form-field">
                <label htmlFor="password" className="form-label">
                    Password <span aria-label="required">*</span>
                </label>
                <input
                    id="password"
                    type="password"
                    className={{`form-input ${{errors.password ? 'error' : ''}}`}}
                    {{...register('password')}}
                    aria-invalid={{errors.password ? 'true' : 'false'}}
                    aria-describedby={{errors.password ? 'password-error' : undefined}}
                />
                {{errors.password && (
                    <p id="password-error" className="form-error" role="alert">
                        {{errors.password.message}}
                    </p>
                )}}
            </div>

            {{/* Confirm Password Field */}}
            <div className="form-field">
                <label htmlFor="confirmPassword" className="form-label">
                    Confirm Password <span aria-label="required">*</span>
                </label>
                <input
                    id="confirmPassword"
                    type="password"
                    className={{`form-input ${{errors.confirmPassword ? 'error' : ''}}`}}
                    {{...register('confirmPassword')}}
                    aria-invalid={{errors.confirmPassword ? 'true' : 'false'}}
                    aria-describedby={{errors.confirmPassword ? 'confirmPassword-error' : undefined}}
                />
                {{errors.confirmPassword && (
                    <p id="confirmPassword-error" className="form-error" role="alert">
                        {{errors.confirmPassword.message}}
                    </p>
                )}}
            </div>

            {{/* Age Field */}}
            <div className="form-field">
                <label htmlFor="age" className="form-label">
                    Age <span aria-label="required">*</span>
                </label>
                <input
                    id="age"
                    type="number"
                    className={{`form-input ${{errors.age ? 'error' : ''}}`}}
                    {{...register('age', {{ valueAsNumber: true }})}}
                    aria-invalid={{errors.age ? 'true' : 'false'}}
                    aria-describedby={{errors.age ? 'age-error' : undefined}}
                />
                {{errors.age && (
                    <p id="age-error" className="form-error" role="alert">
                        {{errors.age.message}}
                    </p>
                )}}
            </div>

            {{/* Terms Checkbox */}}
            <div className="form-field">
                <label className="form-checkbox-label">
                    <input
                        type="checkbox"
                        {{...register('terms')}}
                        aria-invalid={{errors.terms ? 'true' : 'false'}}
                        aria-describedby={{errors.terms ? 'terms-error' : undefined}}
                    />
                    <span>
                        I accept the terms and conditions <span aria-label="required">*</span>
                    </span>
                </label>
                {{errors.terms && (
                    <p id="terms-error" className="form-error" role="alert">
                        {{errors.terms.message}}
                    </p>
                )}}
            </div>

            {{/* Form-level error */}}
            {{submitError && (
                <div className="form-error-banner" role="alert">
                    {{submitError}}
                </div>
            )}}

            {{/* Success message */}}
            {{submitSuccess && (
                <div className="form-success-banner" role="status">
                    Form submitted successfully!
                </div>
            )}}

            {{/* Submit Button */}}
            <button
                type="submit"
                className="form-submit"
                disabled={{isSubmitting || !isDirty || !isValid}}
                aria-busy={{isSubmitting}}
            >
                {{isSubmitting ? 'Submitting...' : 'Submit'}}
            </button>
        </form>
    );
}};
```

VALIDATION BEST PRACTICES:
- Validate on blur for better UX
- Show errors only after user interaction
- Provide clear, actionable error messages
- Use ARIA attributes for accessibility
- Disable submit until form is valid
- Handle loading states
- Server-side validation as backup
- Custom validation rules
- Async validation (e.g., check username availability)
- Field masking for formatted inputs
- Auto-save for long forms

Generate complete, accessible form with validation.""",
    format=PromptFormat.MARKDOWN,
    variables=["form_name", "fields", "validation_rules", "submission"]
)


# Web Security Implementation Prompt
WEB_SECURITY_PROMPT = PromptTemplate(
    template_id="web_security",
    name="Web Security Implementation Prompt",
    template_text="""Implement web security measures.

SECURITY AREAS: {security_areas}
THREAT MODEL: {threat_model}
COMPLIANCE: {compliance}

REQUIREMENTS:
1. XSS (Cross-Site Scripting) prevention
2. CSRF (Cross-Site Request Forgery) protection
3. SQL Injection prevention
4. Content Security Policy (CSP)
5. Security headers
6. Input sanitization
7. Output encoding
8. Rate limiting
9. DDoS protection
10. Secure session management

COMPREHENSIVE SECURITY IMPLEMENTATION:
```python
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import secrets
from typing import Callable
from datetime import datetime, timedelta
import hashlib

app = FastAPI()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration (restrictive)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://www.yourdomain.com"
    ],  # Whitelist specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Specific methods only
    allow_headers=["Content-Type", "Authorization"],  # Specific headers only
    max_age=3600  # Cache preflight requests
)

# Trusted hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["yourdomain.com", "www.yourdomain.com"]
)

# Session middleware with secure settings
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY"),  # From environment
    session_cookie="session_id",
    max_age=3600,  # 1 hour
    same_site="strict",  # CSRF protection
    https_only=True  # Only send over HTTPS
)

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    \"\"\"Add security headers to all responses.\"\"\"

    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)

        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://trusted-cdn.com; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )

        # Prevent XSS attacks
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # HSTS (HTTP Strict Transport Security)
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )

        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions Policy (formerly Feature-Policy)
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        return response


app.add_middleware(SecurityHeadersMiddleware)


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    \"\"\"CSRF token validation for state-changing operations.\"\"\"

    SAFE_METHODS = {{"GET", "HEAD", "OPTIONS", "TRACE"}}

    async def dispatch(self, request: Request, call_next: Callable):
        # Skip CSRF check for safe methods
        if request.method in self.SAFE_METHODS:
            return await call_next(request)

        # Skip CSRF check for API endpoints with token auth
        if request.url.path.startswith("/api/") and "Authorization" in request.headers:
            return await call_next(request)

        # Get CSRF token from header or form
        csrf_token = request.headers.get("X-CSRF-Token") or \
                    (await request.form()).get("csrf_token")

        # Get session token
        session_token = request.session.get("csrf_token")

        if not csrf_token or not session_token or csrf_token != session_token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token validation failed"
            )

        return await call_next(request)


app.add_middleware(CSRFProtectionMiddleware)


def generate_csrf_token() -> str:
    \"\"\"Generate cryptographically secure CSRF token.

    Returns:
        CSRF token string
    \"\"\"
    return secrets.token_urlsafe(32)


def sanitize_input(input_string: str) -> str:
    \"\"\"Sanitize user input to prevent XSS.

    Args:
        input_string: Raw user input

    Returns:
        Sanitized input
    \"\"\"
    import html

    # HTML escape
    sanitized = html.escape(input_string)

    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', '/', '\\\\']
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')

    return sanitized.strip()


def validate_and_sanitize_sql_input(input_value: str) -> Optional[str]:
    \"\"\"Validate and sanitize input for SQL queries.

    Args:
        input_value: Input to validate

    Returns:
        Sanitized input or None if invalid
    \"\"\"
    # Always use parameterized queries
    # This function is for additional validation

    # Check for SQL injection patterns
    sql_patterns = [
        '--', ';--', '/*', '*/', 'xp_', 'sp_', 'exec', 'execute',
        'union', 'select', 'insert', 'update', 'delete', 'drop', 'create',
        'alter', 'declare', 'cast', 'convert'
    ]

    lower_input = input_value.lower()
    for pattern in sql_patterns:
        if pattern in lower_input:
            return None

    return input_value


@app.get("/csrf-token")
async def get_csrf_token(request: Request) -> Dict[str, str]:
    \"\"\"Get CSRF token for form submission.

    Args:
        request: HTTP request

    Returns:
        CSRF token
    \"\"\"
    if "csrf_token" not in request.session:
        request.session["csrf_token"] = generate_csrf_token()

    return {{"csrf_token": request.session["csrf_token"]}}


@app.post("/api/sensitive-action")
@limiter.limit("5/minute")  # Rate limit: 5 requests per minute
async def sensitive_action(
    request: Request,
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    \"\"\"Perform sensitive action with security measures.

    Args:
        request: HTTP request
        data: Action data
        current_user: Authenticated user

    Returns:
        Action result
    \"\"\"
    # Input validation
    if not data or not isinstance(data, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input"
        )

    # Sanitize all string inputs
    sanitized_data = {{
        key: sanitize_input(value) if isinstance(value, str) else value
        for key, value in data.items()
    }}

    # Audit logging
    logger.info(
        "Sensitive action performed",
        extra={{
            "user_id": current_user.get("id"),
            "action": "sensitive_action",
            "ip": request.client.host,
            "user_agent": request.headers.get("user-agent"),
            "timestamp": datetime.utcnow().isoformat()
        }}
    )

    # Perform action (with parameterized queries, not string concatenation)
    result = await db.execute_safe_query(
        "INSERT INTO actions (user_id, data) VALUES (:user_id, :data)",
        {{"user_id": current_user.get("id"), "data": sanitized_data}}
    )

    return {{"success": True, "result": result}}


# Content Security Policy violation reporting
@app.post("/csp-violation-report")
async def csp_violation_report(request: Request):
    \"\"\"Receive and log CSP violation reports.

    Args:
        request: HTTP request with CSP violation data
    \"\"\"
    # Parse JSON safely
    # Assumes request.json() is wrapped or we use a safe helper
    # violation_data = await request.json()
    
    # In zero-error, we might use a helper:
    violation_data, error = await safe_parse_json(request)
    
    if error:
        logger.error(f"Failed to process CSP violation report: {{error}}")
        return {{"status": "error", "detail": "Invalid JSON"}}

    logger.warning(
        "CSP violation detected",
        extra={{"violation": violation_data}}
    )

    return {{"status": "ok"}}
```

SECURITY CHECKLIST:
 HTTPS only (no HTTP)
 Security headers (CSP, HSTS, X-Frame-Options)
 CSRF protection for state-changing operations
 XSS prevention (input sanitization, output encoding)
 SQL injection prevention (parameterized queries)
 Rate limiting on sensitive endpoints
 Authentication and authorization
 Secure session management
 Input validation and sanitization
 Audit logging
 Error handling (no sensitive data in errors)
 Dependency scanning
 Regular security updates
 Penetration testing

Generate complete security implementation.""",
    format=PromptFormat.MARKDOWN,
    variables=["security_areas", "threat_model", "compliance"]
)


# Export all templates
ALL_WEB_TEMPLATES = {
    "react_component": REACT_COMPONENT_PROMPT,
    "rest_api_endpoint": REST_API_PROMPT,
    "authentication_system": AUTHENTICATION_SYSTEM_PROMPT,
    "form_validation": FORM_VALIDATION_PROMPT,
    "web_security": WEB_SECURITY_PROMPT
}


def get_web_template(template_id: str) -> Optional[PromptTemplate]:
    """
    Get web development prompt template by ID.

    Args:
        template_id: Template identifier

    Returns:
        PromptTemplate if found, None otherwise
    """
    return ALL_WEB_TEMPLATES.get(template_id)


def list_web_templates() -> List[str]:
    """
    List all available web development template IDs.

    Returns:
        List of template IDs
    """
    return list(ALL_WEB_TEMPLATES.keys())
