import React, {useState, forwardRef, useImperativeHandle} from 'react';
import Button from 'react-bootstrap/Button';

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
    <Button
      variant= {props.variant}
      disabled={isLoading}
      onClick={!isLoading ? handleClick: null}
    >
      {isLoading ? props.loadingLabel : props.label}
    </Button>
  )
});

export default LoadingButton;