import React, {useState, forwardRef, useImperativeHandle} from 'react';
import Button from 'react-bootstrap/Button';
import styled, {keyframes} from 'styled-components';

const bouncing = () => {
  return keyframes`
  50% {
    transform: scale(1.1);
  }
  `
}

const StyledButton = styled(Button)`
  animation-duration: 1s;
  animation-iteration-count: infinite;
  animation-timing-function: forwards;
  animation-name: ${props => props.bounce ? bouncing : null};
`;

const LoadingButton = forwardRef((props, ref) => {
  const [isLoading, setLoading] = useState(false);
  const handleClick = () => {
    props.handleClick();
    setLoading(true);
  };

  useImperativeHandle(ref, () => ({
    doneLoading: () => doneLoading(),
  }));

  const doneLoading = () => {
    setLoading(false);
  };
  
  return (
    <StyledButton
      variant= {props.variant}
      disabled={isLoading}
      onClick={!isLoading ? handleClick: null}
      bounce={!props.gameStarted && !isLoading}
    >
      {isLoading ? props.loadingLabel : props.label}
    </StyledButton>
  )
});

export default LoadingButton;